// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
#include "volume_builder.h"
#include "volume_impl.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

using namespace wp;

namespace {

struct VolumeDesc {
    // NanoVDB buffer either in device or host memory
    void* buffer = nullptr;
    uint64_t size_in_bytes = 0;
    bool owner = false;  // whether the buffer should be deallocated when the volume is destroyed

    pnanovdb_grid_t grid_data {};
    pnanovdb_tree_t tree_data {};

    // Keep a host-side snapshot so accessors use the same validated, stable
    // metadata for host and device volumes.
    std::vector<pnanovdb_gridblindmetadata_t> blind_metadata;

    // CUDA context for this volume (NULL if CPU)
    void* context = nullptr;

    // Rebuildable CUDA volumes keep actual counts on the device. The host descriptor
    // stores capacities so callers can allocate upper-bound output arrays without
    // forcing a device-to-host count copy during graph capture.
    bool rebuildable = false;
    VolumeRebuildCapacities capacities;

    pnanovdb_buf_t as_pnano() const { return pnanovdb_make_buf(static_cast<uint32_t*>(buffer), size_in_bytes); }
};

// Host-side volume descriptors. Maps each CPU/GPU volume buffer address (id) to a CPU desc
std::map<uint64_t, VolumeDesc> g_volume_descriptors;

bool volume_get_descriptor(uint64_t id, const VolumeDesc*& volumeDesc)
{
    if (id == 0)
        return false;

    const auto& iter = g_volume_descriptors.find(id);
    if (iter == g_volume_descriptors.end())
        return false;
    else
        volumeDesc = &iter->second;
    return true;
}

bool volume_exists(const void* id)
{
    const VolumeDesc* volume;
    return volume_get_descriptor((uint64_t)id, volume);
}

// Resolves a relative offset within the grid. On success, data_offset is no greater than grid_size.
bool volume_resolve_relative_offset(
    uint64_t base_offset, int64_t relative_offset, uint64_t grid_size, uint64_t& data_offset
)
{
    if (base_offset > grid_size)
        return false;

    if (relative_offset < 0) {
        // Avoid negating INT64_MIN directly; -(x + 1) remains representable.
        const uint64_t magnitude = uint64_t(-(relative_offset + 1)) + 1;
        if (magnitude > base_offset)
            return false;
        data_offset = base_offset - magnitude;
    } else {
        const uint64_t offset = uint64_t(relative_offset);
        if (offset > grid_size - base_offset)
            return false;
        data_offset = base_offset + offset;
    }

    return true;
}

// Returns whether the relative offset resolves within the grid and the requested element range fits.
bool volume_resolve_relative_range(
    uint64_t base_offset, int64_t relative_offset, uint64_t value_count, uint32_t value_size, uint64_t grid_size
)
{
    uint64_t data_offset;
    if (!volume_resolve_relative_offset(base_offset, relative_offset, grid_size, data_offset))
        return false;

    // A zero-sized value occupies no bytes, so any value count fits once the offset resolves.
    // Division avoids overflowing when converting the element count to bytes.
    return value_size == 0 || value_count <= (grid_size - data_offset) / value_size;
}

bool volume_validate_grid_metadata(
    const pnanovdb_grid_t& grid_data, uint64_t buffer_size, uint64_t& metadata_offset, uint64_t& metadata_size
)
{
    // A NanoVDB grid starts with a grid header followed by a tree header.
    const uint64_t minimum_grid_size = sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t);
    if (grid_data.grid_size < minimum_grid_size || grid_data.grid_size > buffer_size) {
        return false;
    }

    // Names cross the C ABI and must terminate within their fixed-size fields.
    if (std::memchr(grid_data.grid_name, '\0', sizeof(grid_data.grid_name)) == nullptr)
        return false;

    // The metadata table offset is relative to the grid and must resolve within it.
    if (grid_data.blind_metadata_offset < 0)
        return false;

    metadata_offset = uint64_t(grid_data.blind_metadata_offset);
    if (metadata_offset > grid_data.grid_size)
        return false;

    const uint64_t available = grid_data.grid_size - metadata_offset;
    // Bound the count before computing the table size to avoid multiplication overflow.
    if (grid_data.blind_metadata_count > available / sizeof(pnanovdb_gridblindmetadata_t))
        return false;

    metadata_size = uint64_t(grid_data.blind_metadata_count) * sizeof(pnanovdb_gridblindmetadata_t);
    return true;
}

bool volume_validate_blind_metadata(
    const pnanovdb_grid_t& grid_data,
    const std::vector<pnanovdb_gridblindmetadata_t>& blind_metadata,
    uint64_t metadata_offset
)
{
    for (uint64_t i = 0; i < blind_metadata.size(); ++i) {
        const pnanovdb_gridblindmetadata_t& metadata = blind_metadata[i];
        if (std::memchr(metadata.name, '\0', sizeof(metadata.name)) == nullptr)
            return false;

        // NanoVDB data offsets are relative to the containing metadata entry.
        const uint64_t entry_offset = metadata_offset + i * sizeof(pnanovdb_gridblindmetadata_t);
        if (!volume_resolve_relative_range(
                entry_offset, metadata.data_offset, metadata.value_count, metadata.value_size, grid_data.grid_size
            )) {
            return false;
        }
    }
    return true;
}

// The validation helpers below treat all serialized tree fields as untrusted. They first prove that each node array
// occupies a valid byte range, then verify that every child offset targets the start of a node in the next level.
// This keeps later NanoVDB traversal from interpreting arbitrary payload bytes as nodes.

// Describes the validated byte extent and legal node starts for one NanoVDB tree level. The validator resolves these
// ranges before inspecting node contents, then uses them to ensure that every child offset lands exactly at the start
// of a node in the next level. Fixed-size nodes use begin, end, and stride; variable-size FpN leaves record each legal
// start in node_offsets instead.
struct VolumeNodeRange {
    uint64_t begin = 0;
    uint64_t end = 0;
    uint32_t stride = 0;
    uint32_t count = 0;
    std::vector<uint64_t> node_offsets;
};

bool volume_resolve_node_begin(uint64_t relative_offset, uint64_t grid_size, uint64_t& node_offset)
{
    // Tree node offsets are relative to the tree header, which immediately follows the grid header. Nodes must begin
    // after that header and retain NanoVDB's data alignment. volume_validate_grid_metadata() has already established
    // that grid_size includes both headers, so subtracting tree_offset below cannot underflow.
    const uint64_t tree_offset = sizeof(pnanovdb_grid_t);
    if (relative_offset < sizeof(pnanovdb_tree_t) || relative_offset > grid_size - tree_offset
        || relative_offset % NANOVDB_DATA_ALIGNMENT != 0) {
        return false;
    }

    node_offset = tree_offset + relative_offset;
    return true;
}

bool volume_resolve_node_range(
    uint64_t relative_offset, uint32_t count, uint32_t stride, uint64_t grid_size, VolumeNodeRange& range
)
{
    if (stride == 0)
        return false;

    range.stride = stride;
    range.count = count;
    if (count == 0)
        return true;

    if (!volume_resolve_node_begin(relative_offset, grid_size, range.begin))
        return false;

    // Bound the count before multiplication so the half-open range [begin, end) cannot overflow the grid.
    if (count > (grid_size - range.begin) / stride)
        return false;

    range.end = range.begin + uint64_t(count) * stride;
    return true;
}

bool volume_resolve_fpn_leaf_range(
    const uint8_t* grid_buffer, uint64_t relative_offset, uint32_t count, uint64_t grid_size, VolumeNodeRange& range
)
{
    range.count = count;
    if (count == 0)
        return true;

    if (!volume_resolve_node_begin(relative_offset, grid_size, range.begin))
        return false;

    // FpN leaves have a 96-byte header followed by 512 values packed at a per-leaf width of 1, 2, 4, 8, or 16 bits.
    // Consequently, the encoded value array occupies bit_width * 64 bytes.
    constexpr uint32_t leaf_header_size = 96;
    constexpr uint32_t minimum_leaf_size = leaf_header_size + 64;
    if (count > (grid_size - range.begin) / minimum_leaf_size)
        return false;

    range.node_offsets.reserve(count);
    uint64_t leaf_offset = range.begin;
    for (uint32_t leaf_index = 0; leaf_index < count; ++leaf_index) {
        if (leaf_header_size > grid_size - leaf_offset)
            return false;

        uint32_t bbox_dif_and_flags;
        std::memcpy(
            &bbox_dif_and_flags, grid_buffer + leaf_offset + PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS,
            sizeof(bbox_dif_and_flags)
        );
        // The top three flag bits store log2(bit_width). NanoVDB supports values from zero through four here.
        const uint32_t value_log_bits = bbox_dif_and_flags >> 29;
        if (value_log_bits > 4)
            return false;

        const uint32_t leaf_size = leaf_header_size + (uint32_t(1) << value_log_bits) * 64;
        if (leaf_size > grid_size - leaf_offset)
            return false;

        range.node_offsets.push_back(leaf_offset);
        leaf_offset += leaf_size;
    }

    range.end = leaf_offset;
    return true;
}

bool volume_node_range_contains(const VolumeNodeRange& range, uint64_t node_offset)
{
    // A child reference must land exactly on a node boundary, not merely somewhere inside a node array. Variable-size
    // FpN leaves use their recorded starts; all other node types can use modular arithmetic with a fixed stride.
    if (range.stride == 0)
        return std::binary_search(range.node_offsets.begin(), range.node_offsets.end(), node_offset);

    return node_offset >= range.begin && node_offset < range.end && (node_offset - range.begin) % range.stride == 0;
}

bool volume_validate_child_reference(
    uint64_t parent_offset, int64_t relative_offset, const VolumeNodeRange& child_range, uint64_t grid_size
)
{
    // Internal-node and root-tile child offsets are signed byte offsets relative to their parent node.
    uint64_t child_offset;
    return volume_resolve_relative_offset(parent_offset, relative_offset, grid_size, child_offset)
        && volume_node_range_contains(child_range, child_offset);
}

bool volume_validate_internal_children(
    const uint8_t* grid_buffer,
    const VolumeNodeRange& parent_range,
    uint32_t child_mask_offset,
    uint32_t table_offset,
    uint32_t table_count,
    uint32_t table_stride,
    const VolumeNodeRange& child_range,
    uint64_t grid_size
)
{
    // A set child-mask bit means that the corresponding table slot begins with an int64_t child offset. Validate both
    // structures against the fixed parent stride before reading any serialized mask or table entry.
    const uint32_t mask_word_count = (table_count + 31) / 32;
    if (table_stride < sizeof(int64_t) || child_mask_offset > parent_range.stride
        || mask_word_count > (parent_range.stride - child_mask_offset) / sizeof(uint32_t)
        || table_offset > parent_range.stride || table_count > (parent_range.stride - table_offset) / table_stride) {
        return false;
    }

    for (uint32_t node_index = 0; node_index < parent_range.count; ++node_index) {
        const uint64_t node_offset = parent_range.begin + uint64_t(node_index) * parent_range.stride;
        for (uint32_t word_index = 0; word_index < mask_word_count; ++word_index) {
            uint32_t child_mask;
            std::memcpy(
                &child_mask, grid_buffer + node_offset + child_mask_offset + word_index * sizeof(uint32_t),
                sizeof(child_mask)
            );
            for (uint32_t bit_index = 0; child_mask != 0; ++bit_index, child_mask >>= 1) {
                if ((child_mask & 1) == 0)
                    continue;

                const uint32_t table_index = word_index * 32 + bit_index;
                int64_t relative_offset;
                std::memcpy(
                    &relative_offset, grid_buffer + node_offset + table_offset + uint64_t(table_index) * table_stride,
                    sizeof(relative_offset)
                );
                if (!volume_validate_child_reference(node_offset, relative_offset, child_range, grid_size))
                    return false;
            }
        }
    }

    return true;
}

bool volume_validate_tree(
    const uint8_t* grid_buffer, const pnanovdb_grid_t& grid_data, const pnanovdb_tree_t& tree_data
)
{
    // The caller validates the grid type and breadth-first flag before this helper uses the type-layout table and
    // assumes the serialized order root, root tiles, upper nodes, lower nodes, leaves, then optional blind metadata.
    const uint32_t grid_type = grid_data.grid_type;
    const uint32_t root_size = PNANOVDB_GRID_TYPE_GET(grid_type, root_size);
    const uint32_t root_tile_size = PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size);
    // Reject incomplete type layouts before reading the root's tile count or a tile's child field.
    if (root_size < PNANOVDB_ROOT_BASE_SIZE || root_tile_size < PNANOVDB_ROOT_TILE_BASE_SIZE)
        return false;

    VolumeNodeRange root_range;
    VolumeNodeRange upper_range;
    VolumeNodeRange lower_range;
    VolumeNodeRange leaf_range;
    // Resolve every declared node array before dereferencing it. FpN leaves require a linear scan because their
    // per-leaf bit widths make them variable-sized; all other node arrays have a type-defined fixed stride.
    if (!volume_resolve_node_range(tree_data.node_offset_root, 1, root_size, grid_data.grid_size, root_range)
        || !volume_resolve_node_range(
            tree_data.node_offset_upper, tree_data.node_count_upper, PNANOVDB_GRID_TYPE_GET(grid_type, upper_size),
            grid_data.grid_size, upper_range
        )
        || !volume_resolve_node_range(
            tree_data.node_offset_lower, tree_data.node_count_lower, PNANOVDB_GRID_TYPE_GET(grid_type, lower_size),
            grid_data.grid_size, lower_range
        )
        || (grid_type == PNANOVDB_GRID_TYPE_FPN
                ? !volume_resolve_fpn_leaf_range(
                      grid_buffer, tree_data.node_offset_leaf, tree_data.node_count_leaf, grid_data.grid_size,
                      leaf_range
                  )
                : !volume_resolve_node_range(
                      tree_data.node_offset_leaf, tree_data.node_count_leaf,
                      PNANOVDB_GRID_TYPE_GET(grid_type, leaf_size), grid_data.grid_size, leaf_range
                  ))) {
        return false;
    }

    uint32_t root_tile_count;
    std::memcpy(&root_tile_count, grid_buffer + root_range.begin + PNANOVDB_ROOT_OFF_TABLE_SIZE, sizeof(uint32_t));
    // Root tiles immediately follow the fixed root data, and their serialized count is not covered by the tree header.
    if (root_tile_count > (grid_data.grid_size - root_range.end) / root_tile_size)
        return false;

    const uint64_t root_tiles_begin = root_range.end;
    uint64_t occupied_end = root_tiles_begin + uint64_t(root_tile_count) * root_tile_size;
    // Each populated breadth-first level must begin at or after the previous level ends. Empty levels are skipped
    // because NanoVDB permits their unused offsets to alias another level. Tree data must end before the declared
    // blind-metadata boundary. Metadata-free legacy grids may use zero as an omitted offset, in which case the grid
    // boundary applies instead.
    const uint64_t tree_boundary = grid_data.blind_metadata_count == 0 && grid_data.blind_metadata_offset == 0
        ? grid_data.grid_size
        : uint64_t(grid_data.blind_metadata_offset);
    const auto validate_range_order = [&occupied_end](const VolumeNodeRange& range) {
        if (range.count == 0)
            return true;
        if (range.begin < occupied_end)
            return false;
        occupied_end = range.end;
        return true;
    };
    if (!validate_range_order(upper_range) || !validate_range_order(lower_range) || !validate_range_order(leaf_range)
        || occupied_end > tree_boundary) {
        return false;
    }

    for (uint32_t tile_index = 0; tile_index < root_tile_count; ++tile_index) {
        const uint64_t tile_offset = root_tiles_begin + uint64_t(tile_index) * root_tile_size;
        int64_t relative_offset;
        std::memcpy(
            &relative_offset, grid_buffer + tile_offset + PNANOVDB_ROOT_TILE_OFF_CHILD, sizeof(relative_offset)
        );
        // A zero offset denotes a value tile; nonzero offsets must target an upper node and are relative to the root.
        if (relative_offset != 0
            && !volume_validate_child_reference(root_range.begin, relative_offset, upper_range, grid_data.grid_size)) {
            return false;
        }
    }

    const uint32_t table_stride = PNANOVDB_GRID_TYPE_GET(grid_type, table_stride);
    return volume_validate_internal_children(
               grid_buffer, upper_range, PNANOVDB_UPPER_OFF_CHILD_MASK,
               PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_table), PNANOVDB_UPPER_TABLE_COUNT, table_stride,
               lower_range, grid_data.grid_size
           )
        && volume_validate_internal_children(
               grid_buffer, lower_range, PNANOVDB_LOWER_OFF_CHILD_MASK,
               PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_table), PNANOVDB_LOWER_TABLE_COUNT, table_stride, leaf_range,
               grid_data.grid_size
        );
}

void volume_add_descriptor(uint64_t id, VolumeDesc&& volumeDesc) { g_volume_descriptors[id] = std::move(volumeDesc); }

void volume_rem_descriptor(uint64_t id) { g_volume_descriptors.erase(id); }

void volume_copy_live_metadata(const VolumeDesc* volume, pnanovdb_grid_t& grid_data, pnanovdb_tree_t& tree_data)
{
    if (volume->context) {
        ContextGuard guard(volume->context);
        void* stream = wp_cuda_stream_get_current();
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, &grid_data, volume->buffer, sizeof(pnanovdb_grid_t), stream);
        wp_memcpy_d2h(
            WP_CURRENT_CONTEXT, &tree_data, static_cast<pnanovdb_grid_t*>(volume->buffer) + 1, sizeof(pnanovdb_tree_t),
            stream
        );
        wp_cuda_stream_synchronize(stream);
    } else {
        std::memcpy(&grid_data, volume->buffer, sizeof(pnanovdb_grid_t));
        std::memcpy(&tree_data, static_cast<pnanovdb_grid_t*>(volume->buffer) + 1, sizeof(pnanovdb_tree_t));
    }
}

void volume_mark_rebuildable(uint64_t id, const VolumeRebuildCapacities& capacities)
{
    auto iter = g_volume_descriptors.find(id);
    if (iter == g_volume_descriptors.end())
        return;

    iter->second.rebuildable = true;
    iter->second.capacities = capacities;
}

void volume_set_host_status(uint32_t* status, uint32_t value)
{
    if (status) {
        *status = value;
    }
}

VolumeRebuildCapacities
volume_tile_rebuild_capacities(uint32_t max_tiles, uint32_t max_lower_nodes, uint32_t max_upper_nodes)
{
    VolumeRebuildCapacities capacities;
    capacities.leaf_count = max_tiles;
    capacities.lower_count = max_lower_nodes ? max_lower_nodes : max_tiles;
    capacities.upper_count = max_upper_nodes ? max_upper_nodes : capacities.lower_count;
    capacities.voxel_count = uint64_t(capacities.leaf_count) * PNANOVDB_LEAF_TABLE_COUNT;
    return capacities;
}

VolumeRebuildCapacities volume_voxel_rebuild_capacities(
    uint32_t max_active_voxels, uint32_t max_leaf_nodes, uint32_t max_lower_nodes, uint32_t max_upper_nodes
)
{
    VolumeRebuildCapacities capacities;
    capacities.voxel_count = max_active_voxels;
    capacities.leaf_count = max_leaf_nodes ? max_leaf_nodes : max_active_voxels;
    capacities.lower_count = max_lower_nodes ? max_lower_nodes : capacities.leaf_count;
    capacities.upper_count = max_upper_nodes ? max_upper_nodes : capacities.lower_count;
    return capacities;
}

void volume_set_map(nanovdb::Map& map, const float transform[9], const float translation[3])
{
    // Need to transpose as Map::set is transposing again
    const mat_t<3, 3, double> transpose(
        transform[0], transform[3], transform[6], transform[1], transform[4], transform[7], transform[2], transform[5],
        transform[8]
    );
    const mat_t<3, 3, double> inv = inverse(transpose);

    map.set(transpose.data, inv.data, translation);
}

}  // anonymous namespace

static int volume_validate_grid(const uint8_t* grid_buffer, uint64_t available_size, uint64_t& grid_size)
{
    // Validate one grid from a host buffer. grid_size is returned only after the header and metadata table are bounded,
    // allowing wp_volume_validate_host() to advance safely through files containing concatenated grids.
    if (available_size < sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t))
        return WP_VOLUME_VALIDATION_INVALID;

    pnanovdb_grid_t grid_data;
    pnanovdb_tree_t tree_data;
    std::memcpy(&grid_data, grid_buffer, sizeof(grid_data));
    std::memcpy(&tree_data, grid_buffer + sizeof(grid_data), sizeof(tree_data));

    if (grid_data.magic != PNANOVDB_MAGIC_NUMBER && grid_data.magic != PNANOVDB_MAGIC_GRID)
        return WP_VOLUME_VALIDATION_INVALID;
    if (grid_data.grid_size % NANOVDB_DATA_ALIGNMENT != 0)
        return WP_VOLUME_VALIDATION_INVALID;

    uint64_t metadata_offset;
    uint64_t metadata_size;
    if (!volume_validate_grid_metadata(grid_data, available_size, metadata_offset, metadata_size))
        return WP_VOLUME_VALIDATION_INVALID;

    if (grid_data.grid_type == PNANOVDB_GRID_TYPE_UNKNOWN || grid_data.grid_type >= PNANOVDB_GRID_TYPE_END)
        return WP_VOLUME_VALIDATION_INVALID;

    std::vector<pnanovdb_gridblindmetadata_t> blind_metadata(grid_data.blind_metadata_count);
    if (metadata_size > 0) {
        std::memcpy(blind_metadata.data(), grid_buffer + metadata_offset, metadata_size);
    }

    grid_size = grid_data.grid_size;
    if (!volume_validate_blind_metadata(grid_data, blind_metadata, metadata_offset))
        return WP_VOLUME_VALIDATION_INVALID;
    if ((grid_data.flags & PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST) == 0)
        return WP_VOLUME_VALIDATION_UNSUPPORTED_LAYOUT;
    return volume_validate_tree(grid_buffer, grid_data, tree_data) ? WP_VOLUME_VALIDATION_SUCCESS
                                                                   : WP_VOLUME_VALIDATION_INVALID;
}

int wp_volume_validate_host(const void* buf, uint64_t size)
{
    if (buf == nullptr)
        return WP_VOLUME_VALIDATION_INVALID;

    const uint8_t* buffer = static_cast<const uint8_t*>(buf);
    uint64_t grid_offset = 0;
    int validation_result = WP_VOLUME_VALIDATION_SUCCESS;
    // NanoVDB files may concatenate multiple grids. Require validated grid sizes to consume the buffer exactly, and
    // reject an empty input even though the loop itself would otherwise succeed.
    while (grid_offset < size) {
        uint64_t grid_size;
        const int result = volume_validate_grid(buffer + grid_offset, size - grid_offset, grid_size);
        if (result == WP_VOLUME_VALIDATION_INVALID)
            return WP_VOLUME_VALIDATION_INVALID;
        if (result == WP_VOLUME_VALIDATION_UNSUPPORTED_LAYOUT)
            validation_result = WP_VOLUME_VALIDATION_UNSUPPORTED_LAYOUT;
        else if (result != WP_VOLUME_VALIDATION_SUCCESS)
            return WP_VOLUME_VALIDATION_INVALID;

        grid_offset += grid_size;
    }

    return grid_offset == size && grid_offset != 0 ? validation_result : WP_VOLUME_VALIDATION_INVALID;
}

// NB: buf must be a host pointer
uint64_t wp_volume_create_host(void* buf, uint64_t size, bool copy, bool owner)
{
    if (buf == nullptr || (size > 0 && size < sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t)))
        return 0;  // This cannot be a valid NanoVDB grid with data

    if (!copy && volume_exists(buf)) {
        // descriptor already created for this volume
        return 0;
    }

    VolumeDesc volume;
    volume.context = NULL;

    std::memcpy(&volume.grid_data, buf, sizeof(pnanovdb_grid_t));
    std::memcpy(&volume.tree_data, (pnanovdb_grid_t*)buf + 1, sizeof(pnanovdb_tree_t));

    if (volume.grid_data.magic != PNANOVDB_MAGIC_NUMBER && volume.grid_data.magic != PNANOVDB_MAGIC_GRID)
        return 0;

    if (size == 0) {
        size = volume.grid_data.grid_size;
    }

    uint64_t metadata_offset;
    uint64_t metadata_size;
    if (!volume_validate_grid_metadata(volume.grid_data, size, metadata_offset, metadata_size))
        return 0;

    volume.blind_metadata.resize(volume.grid_data.blind_metadata_count);
    if (metadata_size > 0) {
        std::memcpy(volume.blind_metadata.data(), static_cast<uint8_t*>(buf) + metadata_offset, metadata_size);
    }
    if (!volume_validate_blind_metadata(volume.grid_data, volume.blind_metadata, metadata_offset))
        return 0;

    // Copy or alias buffer
    volume.size_in_bytes = size;
    if (copy) {
        volume.buffer = wp_alloc_host(size, "(native:volume)");
        std::memcpy(volume.buffer, buf, size);
        volume.owner = true;
    } else {
        volume.buffer = buf;
        volume.owner = owner;
    }

    uint64_t id = (uint64_t)volume.buffer;

    volume_add_descriptor(id, std::move(volume));

    return id;
}

// NB: buf must be a pointer on the same device
uint64_t wp_volume_create_device(void* context, void* buf, uint64_t size, bool copy, bool owner)
{
    if (buf == nullptr || (size > 0 && size < sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t)))
        return 0;  // This cannot be a valid NanoVDB grid with data

    if (!copy && volume_exists(buf)) {
        // descriptor already created for this volume
        return 0;
    }

    ContextGuard guard(context);

    VolumeDesc volume;
    volume.context = context ? context : wp_cuda_context_get_current();

    void* stream = wp_cuda_stream_get_current();
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, &volume.grid_data, buf, sizeof(pnanovdb_grid_t), stream);
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, &volume.tree_data, (pnanovdb_grid_t*)buf + 1, sizeof(pnanovdb_tree_t), stream);
    wp_cuda_stream_synchronize(stream);

    if (volume.grid_data.magic != PNANOVDB_MAGIC_NUMBER && volume.grid_data.magic != PNANOVDB_MAGIC_GRID)
        return 0;

    if (size == 0) {
        size = volume.grid_data.grid_size;
    }

    uint64_t metadata_offset;
    uint64_t metadata_size;
    if (!volume_validate_grid_metadata(volume.grid_data, size, metadata_offset, metadata_size))
        return 0;

    volume.blind_metadata.resize(volume.grid_data.blind_metadata_count);
    if (metadata_size > 0) {
        wp_memcpy_d2h(
            WP_CURRENT_CONTEXT, volume.blind_metadata.data(), static_cast<uint8_t*>(buf) + metadata_offset,
            metadata_size, stream
        );
        wp_cuda_stream_synchronize(stream);
    }
    if (!volume_validate_blind_metadata(volume.grid_data, volume.blind_metadata, metadata_offset))
        return 0;

    // Copy or alias data buffer
    volume.size_in_bytes = size;
    if (copy) {
        volume.buffer = wp_alloc_device(WP_CURRENT_CONTEXT, size, "(native:volume)");
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, volume.buffer, buf, size);
        volume.owner = true;
    } else {
        volume.buffer = buf;
        volume.owner = owner;
    }

    uint64_t id = (uint64_t)volume.buffer;
    volume_add_descriptor(id, std::move(volume));

    return id;
}

template <typename BuildT>
uint64_t volume_create_rebuildable_device(
    void* context,
    nanovdb::Grid<nanovdb::NanoTree<BuildT>>* grid,
    uint64_t grid_size,
    const VolumeRebuildCapacities& capacities,
    const BuildGridParams<BuildT>& params
)
{
    if (grid == nullptr || grid_size < sizeof(pnanovdb_grid_t) + sizeof(pnanovdb_tree_t) || volume_exists(grid))
        return 0;

    VolumeDesc volume;
    volume.context = context ? context : wp_cuda_context_get_current();
    volume.buffer = grid;
    volume.size_in_bytes = grid_size;
    volume.owner = true;
    volume.rebuildable = true;
    volume.capacities = capacities;

    volume.grid_data.magic = NANOVDB_MAGIC_GRID;
    volume.grid_data.checksum = ~uint64_t(0);
    volume.grid_data.version = pnanovdb_make_version(
        PNANOVDB_MAJOR_VERSION_NUMBER, PNANOVDB_MINOR_VERSION_NUMBER, PNANOVDB_PATCH_VERSION_NUMBER
    );
    volume.grid_data.flags = uint32_t(nanovdb::GridFlags::HasBBox) | uint32_t(nanovdb::GridFlags::IsBreadthFirst);
    volume.grid_data.grid_count = 1u;
    volume.grid_data.grid_size = grid_size;
    static_assert(sizeof(volume.grid_data.map) == sizeof(params.map));
    std::memcpy(&volume.grid_data.map, &params.map, sizeof(params.map));
    const nanovdb::Vec3d voxel_size = params.map.getVoxelSize();
    for (int k = 0; k < 3; ++k)
        volume.grid_data.voxel_size[k] = voxel_size[k];
    volume.grid_data.grid_class = nanovdb::BuildTraits<BuildT>::is_index ? uint32_t(nanovdb::GridClass::IndexGrid)
                                                                         : uint32_t(nanovdb::GridClass::Unknown);
    volume.grid_data.grid_type = uint32_t(nanovdb::toGridType<BuildT>());
    volume.grid_data.blind_metadata_offset = grid_size;

    const uint64_t root_offset = nanovdb::NanoTree<BuildT>::memUsage();
    const uint64_t upper_offset = root_offset + nanovdb::NanoRoot<BuildT>::memUsage(capacities.upper_count);
    const uint64_t lower_offset
        = upper_offset + nanovdb::NanoUpper<BuildT>::memUsage() * uint64_t(capacities.upper_count);
    const uint64_t leaf_offset
        = lower_offset + nanovdb::NanoLower<BuildT>::memUsage() * uint64_t(capacities.lower_count);
    volume.tree_data.node_offset_root = root_offset;
    volume.tree_data.node_offset_upper = upper_offset;
    volume.tree_data.node_offset_lower = lower_offset;
    volume.tree_data.node_offset_leaf = leaf_offset;
    volume.tree_data.node_count_upper = capacities.upper_count;
    volume.tree_data.node_count_lower = capacities.lower_count;
    volume.tree_data.node_count_leaf = capacities.leaf_count;
    volume.tree_data.tile_count_upper = capacities.upper_count;
    volume.tree_data.tile_count_lower = capacities.lower_count;
    volume.tree_data.tile_count_leaf = capacities.leaf_count;
    volume.tree_data.voxel_count = capacities.voxel_count;

    uint64_t id = reinterpret_cast<uint64_t>(grid);
    volume_add_descriptor(id, std::move(volume));
    return id;
}

void wp_volume_get_buffer_info(uint64_t id, void** buf, uint64_t* size)
{
    *buf = 0;
    *size = 0;

    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        *buf = volume->buffer;
        *size = volume->size_in_bytes;
    }
}

void wp_volume_get_voxel_size(uint64_t id, float* dx, float* dy, float* dz)
{
    *dx = *dy = *dz = 0.0f;

    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        *dx = (float)volume->grid_data.voxel_size[0];
        *dy = (float)volume->grid_data.voxel_size[1];
        *dz = (float)volume->grid_data.voxel_size[2];
    }
}

void wp_volume_get_tile_and_voxel_count(uint64_t id, uint32_t& tile_count, uint64_t& voxel_count)
{
    tile_count = 0;
    voxel_count = 0;

    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        if (volume->rebuildable) {
            tile_count = volume->capacities.leaf_count;
            voxel_count = volume->capacities.voxel_count;
            return;
        }

        tile_count = volume->tree_data.node_count_leaf;

        const uint32_t grid_type = volume->grid_data.grid_type;

        switch (grid_type) {
        case PNANOVDB_GRID_TYPE_ONINDEX:
        case PNANOVDB_GRID_TYPE_ONINDEXMASK:
            // number of indexable voxels is number of active voxels
            voxel_count = volume->tree_data.voxel_count;
            break;
        default:
            // all leaf voxels are indexable
            voxel_count = uint64_t(tile_count) * PNANOVDB_LEAF_TABLE_COUNT;
        }
    }
}

void wp_volume_get_active_stats(
    uint64_t id, uint64_t* voxel_count, uint32_t* leaf_count, uint32_t* lower_count, uint32_t* upper_count
)
{
    *voxel_count = 0;
    *leaf_count = 0;
    *lower_count = 0;
    *upper_count = 0;

    const VolumeDesc* volume;
    if (!volume_get_descriptor(id, volume)) {
        return;
    }

    pnanovdb_grid_t grid_data {};
    pnanovdb_tree_t tree_data {};
    volume_copy_live_metadata(volume, grid_data, tree_data);

    *leaf_count = tree_data.node_count_leaf;
    *lower_count = tree_data.node_count_lower;
    *upper_count = tree_data.node_count_upper;

    switch (grid_data.grid_type) {
    case PNANOVDB_GRID_TYPE_ONINDEX:
    case PNANOVDB_GRID_TYPE_ONINDEXMASK:
        *voxel_count = tree_data.voxel_count;
        break;
    default:
        *voxel_count = uint64_t(*leaf_count) * PNANOVDB_LEAF_TABLE_COUNT;
        break;
    }
}

const char* wp_volume_get_grid_info(
    uint64_t id,
    uint64_t* grid_size,
    uint32_t* grid_index,
    uint32_t* grid_count,
    float translation[3],
    float transform[9],
    char type_str[16]
)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        const pnanovdb_grid_t& grid_data = volume->grid_data;
        *grid_count = grid_data.grid_count;
        *grid_index = grid_data.grid_index;
        *grid_size = grid_data.grid_size;

        memcpy(translation, grid_data.map.vecf, sizeof(grid_data.map.vecf));
        memcpy(transform, grid_data.map.matf, sizeof(grid_data.map.matf));

        nanovdb::toStr(type_str, static_cast<nanovdb::GridType>(grid_data.grid_type));
        return reinterpret_cast<const char*>(grid_data.grid_name);
    }

    *grid_size = 0;
    *grid_index = 0;
    *grid_count = 0;
    type_str[0] = 0;

    return nullptr;
}

uint32_t wp_volume_get_blind_data_count(uint64_t id)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        return volume->grid_data.blind_metadata_count;
    }
    return 0;
}

const char* wp_volume_get_blind_data_info(
    uint64_t id, uint32_t data_index, void** buf, uint64_t* value_count, uint32_t* value_size, char type_str[16]
)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume) && data_index < volume->grid_data.blind_metadata_count) {
        const pnanovdb_gridblindmetadata_t& metadata = volume->blind_metadata[data_index];
        *value_count = metadata.value_count;
        *value_size = metadata.value_size;

        nanovdb::toStr(type_str, static_cast<nanovdb::GridType>(metadata.data_type));
        *buf = static_cast<uint8_t*>(volume->buffer) + volume->grid_data.blind_metadata_offset
            + data_index * sizeof(pnanovdb_gridblindmetadata_t) + metadata.data_offset;
        return reinterpret_cast<const char*>(metadata.name);
    }
    *buf = nullptr;
    *value_count = 0;
    *value_size = 0;
    type_str[0] = 0;
    return nullptr;
}

void wp_volume_get_tiles_host(uint64_t id, void* buf)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        uint32_t leaf_count;
        uint64_t voxel_count;
        wp_volume_get_tile_and_voxel_count(id, leaf_count, voxel_count);

        pnanovdb_coord_t* leaf_coords = static_cast<pnanovdb_coord_t*>(buf);
        const pnanovdb_buf_t pnano_buf = volume->as_pnano();
        const uint32_t actual_leaf_count = pnanovdb_tree_get_node_count_leaf(pnano_buf, volume::get_tree(pnano_buf));
        leaf_count = std::min(leaf_count, actual_leaf_count);

        for (uint32_t i = 0; i < leaf_count; ++i) {
            pnanovdb_leaf_handle_t leaf = volume::get_leaf(pnano_buf, i);
            leaf_coords[i] = volume::leaf_origin(pnano_buf, leaf);
        }
    }
}

void wp_volume_get_voxels_host(uint64_t id, void* buf)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        uint32_t leaf_count;
        uint64_t voxel_count;
        wp_volume_get_tile_and_voxel_count(id, leaf_count, voxel_count);

        pnanovdb_coord_t* voxel_coords = static_cast<pnanovdb_coord_t*>(buf);

        const pnanovdb_buf_t pnano_buf = volume->as_pnano();
        const pnanovdb_tree_handle_t tree = volume::get_tree(pnano_buf);
        const uint32_t actual_leaf_count = pnanovdb_tree_get_node_count_leaf(pnano_buf, tree);
        const uint64_t actual_voxel_count = volume::effective_voxel_count(pnano_buf);
        leaf_count = std::min(leaf_count, actual_leaf_count);
        voxel_count = std::min(voxel_count, actual_voxel_count);

        for (uint32_t i = 0; i < leaf_count; ++i) {
            pnanovdb_leaf_handle_t leaf = volume::get_leaf(pnano_buf, i);
            pnanovdb_coord_t leaf_coords = volume::leaf_origin(pnano_buf, leaf);

            for (uint32_t n = 0; n < 512; ++n) {
                pnanovdb_coord_t loc_ijk = volume::leaf_offset_to_local_coord(n);
                pnanovdb_coord_t ijk = {
                    loc_ijk.x + leaf_coords.x,
                    loc_ijk.y + leaf_coords.y,
                    loc_ijk.z + leaf_coords.z,
                };

                const uint64_t index = volume::leaf_voxel_index(pnano_buf, i, ijk);
                if (index < voxel_count) {
                    voxel_coords[index] = ijk;
                }
            }
        }
    }
}

void wp_volume_destroy_host(uint64_t id)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        if (volume->owner) {
            wp_free_host(volume->buffer);
        }
        volume_rem_descriptor(id);
    }
}

uint64_t wp_volume_from_tiles_host(
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    const void* value_ptr,
    uint32_t value_size,
    const char* value_type,
    bool rebuildable,
    uint32_t max_tiles,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    char gridTypeStr[12];
    const VolumeRebuildCapacities capacities
        = volume_tile_rebuild_capacities(max_tiles, max_lower_nodes, max_upper_nodes);

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    nanovdb::toStr(gridTypeStr, nanovdb::toGridType<type>());                                                          \
    if (strncmp(gridTypeStr, value_type, sizeof(gridTypeStr)) == 0)                                                    \
    {                                                                                                                  \
        BuildGridParams<type> params;                                                                                  \
        memcpy(&params.background_value, value_ptr, value_size);                                                       \
        volume_set_map(params.map, transform, translation);                                                            \
        size_t gridSize;                                                                                               \
        nanovdb::Grid<nanovdb::NanoTree<type>>* grid;                                                                  \
        if (rebuildable)                                                                                               \
        {                                                                                                              \
            allocate_rebuildable_grid_from_tiles_host(                                                                 \
                grid, gridSize, points, num_points, points_in_world_space, point_mask, capacities, params, status       \
            );                                                                                                         \
            uint64_t id = wp_volume_create_host(grid, gridSize, false, true);                                          \
            volume_mark_rebuildable(id, capacities);                                                                   \
            return id;                                                                                                 \
        }                                                                                                              \
        allocate_grid_from_tiles_host(grid, gridSize, points, num_points, points_in_world_space, point_mask, params);   \
        return wp_volume_create_host(grid, gridSize, false, true);                                                     \
    }

    WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

    return 0;
}

uint64_t wp_volume_index_from_tiles_host(
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    bool rebuildable,
    uint32_t max_tiles,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    nanovdb::IndexGrid* grid;
    size_t gridSize;
    BuildGridParams<nanovdb::ValueIndex> params;
    volume_set_map(params.map, transform, translation);

    if (rebuildable) {
        const VolumeRebuildCapacities capacities
            = volume_tile_rebuild_capacities(max_tiles, max_lower_nodes, max_upper_nodes);
        allocate_rebuildable_grid_from_tiles_host(
            grid, gridSize, points, num_points, points_in_world_space, point_mask, capacities, params, status
        );

        uint64_t id = wp_volume_create_host(grid, gridSize, false, true);
        volume_mark_rebuildable(id, capacities);
        return id;
    }

    allocate_grid_from_tiles_host(grid, gridSize, points, num_points, points_in_world_space, point_mask, params);
    return wp_volume_create_host(grid, gridSize, false, true);
}

uint64_t wp_volume_from_active_voxels_host(
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    bool rebuildable,
    uint32_t max_active_voxels,
    uint32_t max_leaf_nodes,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    nanovdb::OnIndexGrid* grid;
    size_t gridSize;
    BuildGridParams<nanovdb::ValueOnIndex> params;
    volume_set_map(params.map, transform, translation);

    if (rebuildable) {
        const VolumeRebuildCapacities capacities
            = volume_voxel_rebuild_capacities(max_active_voxels, max_leaf_nodes, max_lower_nodes, max_upper_nodes);
        allocate_rebuildable_grid_from_active_voxels_host(
            grid, gridSize, points, num_points, points_in_world_space, point_mask, capacities, params, status
        );

        uint64_t id = wp_volume_create_host(grid, gridSize, false, true);
        volume_mark_rebuildable(id, capacities);
        return id;
    }

    allocate_grid_from_active_voxels_host(
        grid, gridSize, points, num_points, points_in_world_space, point_mask, params
    );
    return wp_volume_create_host(grid, gridSize, false, true);
}

void wp_volume_rebuild_from_tiles_host(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    const void* value_ptr,
    uint32_t value_size,
    const char* value_type,
    uint32_t* status
)
{
    const VolumeDesc* volume;
    if (!volume_get_descriptor(id, volume) || !volume->rebuildable) {
        volume_set_host_status(status, WP_VOLUME_REBUILD_INVALID_INPUT);
        return;
    }

    char gridTypeStr[12];

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    nanovdb::toStr(gridTypeStr, nanovdb::toGridType<type>());                                                          \
    if (strncmp(gridTypeStr, value_type, sizeof(gridTypeStr)) == 0)                                                    \
    {                                                                                                                  \
        BuildGridParams<type> params;                                                                                  \
        memcpy(&params.background_value, value_ptr, value_size);                                                       \
        volume_set_map(params.map, transform, translation);                                                            \
        rebuild_grid_from_tiles_host(                                                                                  \
            reinterpret_cast<nanovdb::Grid<nanovdb::NanoTree<type>>*>(volume->buffer), volume->size_in_bytes, points,  \
            num_points, points_in_world_space, point_mask, volume->capacities, params, status                          \
        );                                                                                                             \
        return;                                                                                                        \
    }

    WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

    volume_set_host_status(status, WP_VOLUME_REBUILD_INVALID_INPUT);
}

void wp_volume_index_rebuild_from_tiles_host(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    uint32_t* status
)
{
    const VolumeDesc* volume;
    if (!volume_get_descriptor(id, volume) || !volume->rebuildable) {
        volume_set_host_status(status, WP_VOLUME_REBUILD_INVALID_INPUT);
        return;
    }

    BuildGridParams<nanovdb::ValueIndex> params;
    volume_set_map(params.map, transform, translation);
    rebuild_grid_from_tiles_host(
        reinterpret_cast<nanovdb::IndexGrid*>(volume->buffer), volume->size_in_bytes, points, num_points,
        points_in_world_space, point_mask, volume->capacities, params, status
    );
}

void wp_volume_rebuild_from_active_voxels_host(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    uint32_t* status
)
{
    const VolumeDesc* volume;
    if (!volume_get_descriptor(id, volume) || !volume->rebuildable) {
        volume_set_host_status(status, WP_VOLUME_REBUILD_INVALID_INPUT);
        return;
    }

    BuildGridParams<nanovdb::ValueOnIndex> params;
    volume_set_map(params.map, transform, translation);
    rebuild_grid_from_active_voxels_host(
        reinterpret_cast<nanovdb::OnIndexGrid*>(volume->buffer), volume->size_in_bytes, points, num_points,
        points_in_world_space, point_mask, volume->capacities, params, status
    );
}

void wp_volume_destroy_device(uint64_t id)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        ContextGuard guard(volume->context);
        if (volume->owner) {
            wp_free_device(WP_CURRENT_CONTEXT, volume->buffer);
        }
        volume_rem_descriptor(id);
    }
}

#if WP_ENABLE_CUDA

uint64_t wp_volume_from_tiles_device(
    void* context,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    const void* value_ptr,
    uint32_t value_size,
    const char* value_type,
    bool rebuildable,
    uint32_t max_tiles,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    char gridTypeStr[12];
    const VolumeRebuildCapacities capacities
        = volume_tile_rebuild_capacities(max_tiles, max_lower_nodes, max_upper_nodes);

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    nanovdb::toStr(gridTypeStr, nanovdb::toGridType<type>());                                                          \
    if (strncmp(gridTypeStr, value_type, sizeof(gridTypeStr)) == 0)                                                    \
    {                                                                                                                  \
        BuildGridParams<type> params;                                                                                  \
        memcpy(&params.background_value, value_ptr, value_size);                                                       \
        volume_set_map(params.map, transform, translation);                                                            \
        size_t gridSize;                                                                                               \
        nanovdb::Grid<nanovdb::NanoTree<type>>* grid;                                                                  \
        if (rebuildable)                                                                                               \
        {                                                                                                              \
            allocate_rebuildable_grid_from_tiles(                                                                      \
                grid, gridSize, points, num_points, points_in_world_space, point_mask, capacities, params, status       \
            );                                                                                                         \
            return volume_create_rebuildable_device(context, grid, gridSize, capacities, params);                     \
        }                                                                                                              \
        allocate_grid_from_tiles(grid, gridSize, points, num_points, points_in_world_space, point_mask, params);        \
        return wp_volume_create_device(context, grid, gridSize, false, true);                                           \
    }

    WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

    return 0;
}

uint64_t wp_volume_index_from_tiles_device(
    void* context,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    bool rebuildable,
    uint32_t max_tiles,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    nanovdb::IndexGrid* grid;
    size_t gridSize;
    BuildGridParams<nanovdb::ValueIndex> params;
    volume_set_map(params.map, transform, translation);

    if (rebuildable) {
        const VolumeRebuildCapacities capacities
            = volume_tile_rebuild_capacities(max_tiles, max_lower_nodes, max_upper_nodes);
        allocate_rebuildable_grid_from_tiles(
            grid, gridSize, points, num_points, points_in_world_space, point_mask, capacities, params, status
        );

        return volume_create_rebuildable_device(context, grid, gridSize, capacities, params);
    }

    allocate_grid_from_tiles(grid, gridSize, points, num_points, points_in_world_space, point_mask, params);
    return wp_volume_create_device(context, grid, gridSize, false, true);
}

uint64_t wp_volume_from_active_voxels_device(
    void* context,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    bool rebuildable,
    uint32_t max_active_voxels,
    uint32_t max_leaf_nodes,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    nanovdb::OnIndexGrid* grid;
    size_t gridSize;
    BuildGridParams<nanovdb::ValueOnIndex> params;
    volume_set_map(params.map, transform, translation);

    if (rebuildable) {
        const VolumeRebuildCapacities capacities
            = volume_voxel_rebuild_capacities(max_active_voxels, max_leaf_nodes, max_lower_nodes, max_upper_nodes);
        allocate_rebuildable_grid_from_active_voxels(
            grid, gridSize, points, num_points, points_in_world_space, point_mask, capacities, params, status
        );

        return volume_create_rebuildable_device(context, grid, gridSize, capacities, params);
    }

    allocate_grid_from_active_voxels(grid, gridSize, points, num_points, points_in_world_space, point_mask, params);
    return wp_volume_create_device(context, grid, gridSize, false, true);
}

void wp_volume_rebuild_from_tiles_device(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    const void* value_ptr,
    uint32_t value_size,
    const char* value_type,
    uint32_t* status
)
{
    const VolumeDesc* volume;
    if (!volume_get_descriptor(id, volume) || !volume->rebuildable) {
        if (status) {
            uint32_t invalid = WP_VOLUME_REBUILD_INVALID_INPUT;
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, status, &invalid, sizeof(uint32_t));
        }
        return;
    }

    ContextGuard guard(volume->context);

    char gridTypeStr[12];

#define EXPAND_BUILDER_TYPE(type)                                                                                      \
    nanovdb::toStr(gridTypeStr, nanovdb::toGridType<type>());                                                          \
    if (strncmp(gridTypeStr, value_type, sizeof(gridTypeStr)) == 0)                                                    \
    {                                                                                                                  \
        BuildGridParams<type> params;                                                                                  \
        memcpy(&params.background_value, value_ptr, value_size);                                                       \
        volume_set_map(params.map, transform, translation);                                                            \
        rebuild_grid_from_tiles(                                                                                       \
            reinterpret_cast<nanovdb::Grid<nanovdb::NanoTree<type>>*>(volume->buffer), volume->size_in_bytes, points,  \
            num_points, points_in_world_space, point_mask, volume->capacities, params, status                          \
        );                                                                                                             \
        return;                                                                                                        \
    }

    WP_VOLUME_BUILDER_INSTANTIATE_TYPES
#undef EXPAND_BUILDER_TYPE

    if (status) {
        uint32_t invalid = WP_VOLUME_REBUILD_INVALID_INPUT;
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, status, &invalid, sizeof(uint32_t));
    }
    return;
}

void wp_volume_index_rebuild_from_tiles_device(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    uint32_t* status
)
{
    const VolumeDesc* volume;
    if (!volume_get_descriptor(id, volume) || !volume->rebuildable) {
        if (status) {
            uint32_t invalid = WP_VOLUME_REBUILD_INVALID_INPUT;
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, status, &invalid, sizeof(uint32_t));
        }
        return;
    }

    ContextGuard guard(volume->context);

    BuildGridParams<nanovdb::ValueIndex> params;
    volume_set_map(params.map, transform, translation);
    rebuild_grid_from_tiles(
        reinterpret_cast<nanovdb::IndexGrid*>(volume->buffer), volume->size_in_bytes, points, num_points,
        points_in_world_space, point_mask, volume->capacities, params, status
    );
}

void wp_volume_rebuild_from_active_voxels_device(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    uint32_t* status
)
{
    const VolumeDesc* volume;
    if (!volume_get_descriptor(id, volume) || !volume->rebuildable) {
        if (status) {
            uint32_t invalid = WP_VOLUME_REBUILD_INVALID_INPUT;
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, status, &invalid, sizeof(uint32_t));
        }
        return;
    }

    ContextGuard guard(volume->context);

    BuildGridParams<nanovdb::ValueOnIndex> params;
    volume_set_map(params.map, transform, translation);
    rebuild_grid_from_active_voxels(
        reinterpret_cast<nanovdb::OnIndexGrid*>(volume->buffer), volume->size_in_bytes, points, num_points,
        points_in_world_space, point_mask, volume->capacities, params, status
    );
}

void launch_get_leaf_coords(
    void* context, const uint32_t leaf_count, pnanovdb_coord_t* leaf_coords, pnanovdb_buf_t buf
);
void launch_get_voxel_coords(
    void* context,
    const uint32_t leaf_count,
    const uint32_t voxel_count,
    pnanovdb_coord_t* voxel_coords,
    pnanovdb_buf_t buf
);

void wp_volume_get_tiles_device(uint64_t id, void* buf)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        uint32_t leaf_count;
        uint64_t voxel_count;
        wp_volume_get_tile_and_voxel_count(id, leaf_count, voxel_count);

        pnanovdb_coord_t* leaf_coords = static_cast<pnanovdb_coord_t*>(buf);
        launch_get_leaf_coords(volume->context, leaf_count, leaf_coords, volume->as_pnano());
    }
}

void wp_volume_get_voxels_device(uint64_t id, void* buf)
{
    const VolumeDesc* volume;
    if (volume_get_descriptor(id, volume)) {
        uint32_t leaf_count;
        uint64_t voxel_count;
        wp_volume_get_tile_and_voxel_count(id, leaf_count, voxel_count);

        pnanovdb_coord_t* voxel_coords = static_cast<pnanovdb_coord_t*>(buf);
        launch_get_voxel_coords(volume->context, leaf_count, voxel_count, voxel_coords, volume->as_pnano());
    }
}

#else
// stubs for non-CUDA platforms
uint64_t wp_volume_from_tiles_device(
    void* context,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    const void* value_ptr,
    uint32_t value_size,
    const char* value_type,
    bool rebuildable,
    uint32_t max_tiles,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    return 0;
}

uint64_t wp_volume_index_from_tiles_device(
    void* context,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    bool rebuildable,
    uint32_t max_tiles,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    return 0;
}

uint64_t wp_volume_from_active_voxels_device(
    void* context,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    bool rebuildable,
    uint32_t max_active_voxels,
    uint32_t max_leaf_nodes,
    uint32_t max_lower_nodes,
    uint32_t max_upper_nodes,
    uint32_t* status
)
{
    return 0;
}

void wp_volume_rebuild_from_tiles_device(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    const void* value_ptr,
    uint32_t value_size,
    const char* value_type,
    uint32_t* status
)
{
}

void wp_volume_index_rebuild_from_tiles_device(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    uint32_t* status
)
{
}

void wp_volume_rebuild_from_active_voxels_device(
    uint64_t id,
    void* points,
    int num_points,
    const int32_t* point_mask,
    float transform[9],
    float translation[3],
    bool points_in_world_space,
    uint32_t* status
)
{
}

void wp_volume_get_tiles_device(uint64_t id, void* buf) { }

void wp_volume_get_voxels_device(uint64_t id, void* buf) { }

#endif

const char* wp_nanovdb_version()
{
    static char version[64];
    snprintf(
        version, sizeof(version), "%d.%d.%d", PNANOVDB_MAJOR_VERSION_NUMBER, PNANOVDB_MINOR_VERSION_NUMBER,
        PNANOVDB_PATCH_VERSION_NUMBER
    );
    return version;
}
