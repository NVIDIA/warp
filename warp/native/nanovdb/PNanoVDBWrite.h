
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file   PNanoVDBWrite.h

	\author Andrew Reidmeyer

	\brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
			of NanoVDBWrite.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PNANOVDB_WRITE_H_HAS_BEEN_INCLUDED
#define NANOVDB_PNANOVDB_WRITE_H_HAS_BEEN_INCLUDED

#if defined(PNANOVDB_BUF_C)
#if defined(PNANOVDB_ADDRESS_32)
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint32_t byte_offset, uint32_t value)
{
	uint32_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	if (wordaddress < buf.size_in_words)
	{
		buf.data[wordaddress] = value;
	}
#else
	buf.data[wordaddress] = value;
#endif
}
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint32_t byte_offset, uint64_t value)
{
	uint64_t* data64 = (uint64_t*)buf.data;
	uint32_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	uint64_t size_in_words64 = buf.size_in_words >> 1u;
	if (wordaddress64 < size_in_words64)
	{
		data64[wordaddress64] = value;
	}
#else
	data64[wordaddress64] = value;
#endif
}
#elif defined(PNANOVDB_ADDRESS_64)
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint64_t byte_offset, uint32_t value)
{
	uint64_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	if (wordaddress < buf.size_in_words)
	{
		buf.data[wordaddress] = value;
	}
#else
	buf.data[wordaddress] = value;
#endif
}
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint64_t byte_offset, uint64_t value)
{
	uint64_t* data64 = (uint64_t*)buf.data;
	uint64_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	uint64_t size_in_words64 = buf.size_in_words >> 1u;
	if (wordaddress64 < size_in_words64)
	{
		data64[wordaddress64] = value;
	}
#else
	data64[wordaddress64] = value;
#endif
}
#endif
#endif

#if defined(PNANOVDB_C)
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_float_as_uint32(float v) { return *((pnanovdb_uint32_t*)(&v)); }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { return *((pnanovdb_uint64_t*)(&v)); }
#elif defined(PNANOVDB_HLSL)
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_float_as_uint32(float v) { return asuint(v); }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { uint2 ret; asuint(v, ret.x, ret.y); return ret; }
#elif defined(PNANOVDB_GLSL)
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_float_as_uint32(float v) { return floatBitsToUint(v); }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { return unpackDouble2x32(v); }
#endif

PNANOVDB_FORCE_INLINE void pnanovdb_write_uint32(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_uint32_t value)
{
	pnanovdb_buf_write_uint32(buf, address.byte_offset, value);
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_uint64(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_uint64_t value)
{
	pnanovdb_buf_write_uint64(buf, address.byte_offset, value);
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_int32(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_int32_t value)
{
	pnanovdb_write_uint32(buf, address, pnanovdb_int32_as_uint32(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_int64(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_int64_t value)
{
	pnanovdb_buf_write_uint64(buf, address.byte_offset, pnanovdb_int64_as_uint64(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_float(pnanovdb_buf_t buf, pnanovdb_address_t address, float value)
{
	pnanovdb_write_uint32(buf, address, pnanovdb_float_as_uint32(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_double(pnanovdb_buf_t buf, pnanovdb_address_t address, double value)
{
	pnanovdb_write_uint64(buf, address, pnanovdb_double_as_uint64(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_coord(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) value)
{
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 0u), pnanovdb_int32_as_uint32(PNANOVDB_DEREF(value).x));
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 4u), pnanovdb_int32_as_uint32(PNANOVDB_DEREF(value).y));
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 8u), pnanovdb_int32_as_uint32(PNANOVDB_DEREF(value).z));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_vec3(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_vec3_t) value)
{
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 0u), pnanovdb_float_as_uint32(PNANOVDB_DEREF(value).x));
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 4u), pnanovdb_float_as_uint32(PNANOVDB_DEREF(value).y));
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 8u), pnanovdb_float_as_uint32(PNANOVDB_DEREF(value).z));
}

PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_leaf) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF), node_offset_leaf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_lower) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER), node_offset_lower);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_upper) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER), node_offset_upper);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_root(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_root) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT), node_offset_root);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_count_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t node_count_leaf) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_LEAF), node_count_leaf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_count_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t node_count_lower) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_LOWER), node_count_lower);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_count_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t node_count_upper) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_UPPER), node_count_upper);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_tile_count_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t tile_count_leaf) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_LEAF), tile_count_leaf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_tile_count_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t tile_count_lower) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_LOWER), tile_count_lower);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_tile_count_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t tile_count_upper) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_UPPER), tile_count_upper);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_voxel_count(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t voxel_count) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_VOXEL_COUNT), voxel_count);
}

PNANOVDB_FORCE_INLINE void pnanovdb_root_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_root_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
	pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_set_bbox_max(pnanovdb_buf_t buf, pnanovdb_root_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_max) {
	pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MAX), bbox_max);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_set_tile_count(pnanovdb_buf_t buf, pnanovdb_root_handle_t p, pnanovdb_uint32_t tile_count) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_TABLE_SIZE), tile_count);
}

PNANOVDB_FORCE_INLINE void pnanovdb_root_tile_set_key(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p, pnanovdb_uint64_t key) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_KEY), key);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_tile_set_child(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p, pnanovdb_int64_t child) {
	pnanovdb_write_int64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_CHILD), child);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_tile_set_state(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p, pnanovdb_uint32_t state) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_STATE), state);
}

PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
	pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_bbox_max(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_max) {
	pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_BBOX_MAX), bbox_max);
}
PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_child_mask(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, pnanovdb_uint32_t bit_index, pnanovdb_bool_t value) {
	pnanovdb_address_t addr = pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_CHILD_MASK + 4u * (bit_index >> 5u));
	pnanovdb_uint32_t valueMask = pnanovdb_read_uint32(buf, addr);
	if (!value) { valueMask &= ~(1u << (bit_index & 31u)); }
	if (value) valueMask |= (1u << (bit_index & 31u));
	pnanovdb_write_uint32(buf, addr, valueMask);
}
PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_table_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node, pnanovdb_uint32_t n, pnanovdb_int64_t child)
{
	pnanovdb_address_t bufAddress = pnanovdb_upper_get_table_address(grid_type, buf, node, n);
	pnanovdb_write_int64(buf, bufAddress, child);
}

PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
	pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_bbox_max(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_max) {
	pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_BBOX_MAX), bbox_max);
}
PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_child_mask(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, pnanovdb_uint32_t bit_index, pnanovdb_bool_t value) {
	pnanovdb_address_t addr = pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_CHILD_MASK + 4u * (bit_index >> 5u));
	pnanovdb_uint32_t valueMask = pnanovdb_read_uint32(buf, addr);
	if (!value) { valueMask &= ~(1u << (bit_index & 31u)); }
	if (value) valueMask |= (1u << (bit_index & 31u));
	pnanovdb_write_uint32(buf, addr, valueMask);
}
PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_table_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node, pnanovdb_uint32_t n, pnanovdb_int64_t child)
{
	pnanovdb_address_t table_address = pnanovdb_lower_get_table_address(grid_type, buf, node, n);
	pnanovdb_write_int64(buf, table_address, child);
}

PNANOVDB_FORCE_INLINE void pnanovdb_leaf_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
	pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LEAF_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_leaf_set_bbox_dif_and_flags(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t p, pnanovdb_uint32_t bbox_dif_and_flags) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS), bbox_dif_and_flags);
}

PNANOVDB_FORCE_INLINE void pnanovdb_map_set_matf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float matf) {
	pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATF + 4u * index), matf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_invmatf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float invmatf) {
	pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATF + 4u * index), invmatf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_vecf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float vecf) {
	pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECF + 4u * index), vecf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_taperf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float taperf) {
	pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERF), taperf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_matd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double matd) {
	pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATD + 8u * index), matd);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_invmatd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double invmatd) {
	pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATD + 8u * index), invmatd);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_vecd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double vecd) {
	pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECD + 8u * index), vecd);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_taperd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double taperd) {
	pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERD), taperd);
}

PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_magic(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t magic) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_MAGIC), magic);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_checksum(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t checksum) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_CHECKSUM), checksum);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_version(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t version) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VERSION), version);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_flags(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t flags) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_FLAGS), flags);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_get_grid_index(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_index) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_INDEX), grid_index);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_get_grid_count(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_count) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_COUNT), grid_count);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t grid_size) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_SIZE), grid_size);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_name(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index, pnanovdb_uint32_t grid_name) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_NAME + 4u * index), grid_name);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_world_bbox(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index, double world_bbox) {
	pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_WORLD_BBOX + 8u * index), world_bbox);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_voxel_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index, double voxel_size) {
	pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VOXEL_SIZE + 8u * index), voxel_size);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_class(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_class) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_CLASS), grid_class);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_type(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_type) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_TYPE), grid_type);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_blind_metadata_offset(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t blind_metadata_offset) {
	pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET), blind_metadata_offset);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_blind_metadata_count(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t metadata_count) {
	pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT), metadata_count);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_make_version(pnanovdb_uint32_t major, pnanovdb_uint32_t minor, pnanovdb_uint32_t patch)
{
	return (major << 21u) | (minor << 10u) | (patch);
}

#endif