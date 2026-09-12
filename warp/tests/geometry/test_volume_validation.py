# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NanoVDB volume structure and metadata validation."""

import os
import struct
import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Byte offsets mirror the corresponding PNanoVDB.h layout macros.
_PNANOVDB_GRID_OFF_GRID_SIZE = 32
_PNANOVDB_GRID_OFF_GRID_NAME = 40
_PNANOVDB_GRID_OFF_GRID_TYPE = 636
_PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET = 640
_PNANOVDB_GRID_OFF_FLAGS = 20
_PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST = 1 << 5
_PNANOVDB_GRIDBLINDMETADATA_OFF_NAME = 32
_PNANOVDB_GRID_SIZE = 672
_PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF = 0
_PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER = 8
_PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER = 16
_PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT = 24
_PNANOVDB_TREE_OFF_NODE_COUNT_LEAF = 32
_PNANOVDB_TREE_OFF_NODE_COUNT_LOWER = 36
_PNANOVDB_TREE_OFF_NODE_COUNT_UPPER = 40
_PNANOVDB_ROOT_OFF_TABLE_SIZE = 24
# FpN is NanoVDB's variable-bit quantized floating-point grid type.
_PNANOVDB_ROOT_SIZE_FPN = 64
_PNANOVDB_ROOT_SIZE_INDEX = 96
_PNANOVDB_ROOT_TILE_OFF_CHILD = 8
_PNANOVDB_ROOT_TILE_SIZE_INDEX = 32
_PNANOVDB_UPPER_OFF_CHILD_MASK = 4128
_PNANOVDB_UPPER_OFF_TABLE_INDEX = 8256
_PNANOVDB_UPPER_SIZE_INDEX = 270400
_PNANOVDB_UPPER_TABLE_COUNT = 32768
_PNANOVDB_LOWER_OFF_CHILD_MASK = 544
_PNANOVDB_LOWER_OFF_TABLE_INDEX = 1088
_PNANOVDB_LOWER_SIZE_INDEX = 33856
_PNANOVDB_LOWER_TABLE_COUNT = 4096
_PNANOVDB_GRID_TYPE_FPN = 16
_PNANOVDB_NAME_SIZE = 256

_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
# This fixture contains two concatenated index grids; its first grid includes blind metadata used by mutation tests.
_INDEX_GRID_PATH = os.path.join(_ASSET_DIR, "test_index_grid.nvdb")
# These legacy float, int32, and vec3f grids have no blind metadata and encode the unused metadata offset as zero.
_METADATA_FREE_GRID_PATHS = tuple(
    os.path.join(_ASSET_DIR, name) for name in ("test_grid.nvdb", "test_int32_grid.nvdb", "test_vec_grid.nvdb")
)

_index_grid_data = None


def _get_index_grid_data():
    """Load and cache the decompressed index-grid fixture."""
    global _index_grid_data

    if _index_grid_data is None:
        with open(_INDEX_GRID_PATH, "rb") as stream:
            volume = wp.Volume.load_from_nvdb(stream, device="cpu")
        _index_grid_data = volume.array().numpy().copy()

    return _index_grid_data.copy()


def _assert_volume_rejected(test, device, grid_data):
    data = wp.array(grid_data, dtype=wp.uint8, device=device)
    with test.assertRaises(RuntimeError):
        wp.Volume(data)


def _serialize_nvdb_grids(*grids):
    """Serialize ``grids`` as an uncompressed NanoVDB file buffer."""
    file_header = struct.pack("<QIHH", 0x304244566F6E614E, 32 << 21 | 3 << 10 | 3, len(grids), 0)
    metadata = bytearray()
    payload = bytearray()
    for grid_data in grids:
        file_metadata = bytearray(176)
        struct.pack_into("<QQ", file_metadata, 0, len(grid_data), len(grid_data))
        struct.pack_into("<I", file_metadata, 136, 1)
        metadata += file_metadata + b"\0"
        payload += bytes(grid_data)
    return file_header + metadata + payload


def _serialize_nvdb_grid(grid_data):
    return _serialize_nvdb_grids(grid_data)


def _find_index_grid_child_offsets(grid_data):
    """Find the serialized child-reference fields for each index-grid tree level."""
    tree_offset = _PNANOVDB_GRID_SIZE
    root_offset = struct.unpack_from("<Q", grid_data, tree_offset + _PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT)[0]
    root = tree_offset + root_offset
    root_tile_count = struct.unpack_from("<I", grid_data, root + _PNANOVDB_ROOT_OFF_TABLE_SIZE)[0]
    for index in range(root_tile_count):
        child_offset = (
            root + _PNANOVDB_ROOT_SIZE_INDEX + index * _PNANOVDB_ROOT_TILE_SIZE_INDEX + _PNANOVDB_ROOT_TILE_OFF_CHILD
        )
        if struct.unpack_from("<q", grid_data, child_offset)[0] != 0:
            root_child_offset = child_offset
            break
    else:
        raise AssertionError("Index grid has no root child reference")

    def find_masked_child(node_offset_field, node_count_field, node_size, mask_offset, table_offset, table_count):
        """Find the first masked child-reference field in a serialized node level."""
        relative_offset = struct.unpack_from("<Q", grid_data, tree_offset + node_offset_field)[0]
        node_count = struct.unpack_from("<I", grid_data, tree_offset + node_count_field)[0]
        nodes = tree_offset + relative_offset
        for node_index in range(node_count):
            node = nodes + node_index * node_size
            for table_index in range(table_count):
                mask = grid_data[node + mask_offset + table_index // 8]
                if mask & (1 << (table_index % 8)):
                    return node + table_offset + table_index * 8
        raise AssertionError("Index grid has no masked child reference")

    upper_child_offset = find_masked_child(
        _PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER,
        _PNANOVDB_TREE_OFF_NODE_COUNT_UPPER,
        _PNANOVDB_UPPER_SIZE_INDEX,
        _PNANOVDB_UPPER_OFF_CHILD_MASK,
        _PNANOVDB_UPPER_OFF_TABLE_INDEX,
        _PNANOVDB_UPPER_TABLE_COUNT,
    )
    lower_child_offset = find_masked_child(
        _PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER,
        _PNANOVDB_TREE_OFF_NODE_COUNT_LOWER,
        _PNANOVDB_LOWER_SIZE_INDEX,
        _PNANOVDB_LOWER_OFF_CHILD_MASK,
        _PNANOVDB_LOWER_OFF_TABLE_INDEX,
        _PNANOVDB_LOWER_TABLE_COUNT,
    )
    return root_child_offset, upper_child_offset, lower_child_offset


def _make_fpn_grid(leaf_bit_widths):
    """Build an FpN grid with leaves encoded at ``leaf_bit_widths``."""
    root_offset = 64
    upper_offset = root_offset + _PNANOVDB_ROOT_SIZE_FPN + _PNANOVDB_ROOT_TILE_SIZE_INDEX
    lower_offset = upper_offset + _PNANOVDB_UPPER_SIZE_INDEX
    leaf_offset = lower_offset + _PNANOVDB_LOWER_SIZE_INDEX
    leaf_sizes = [96 + bit_width * 64 for bit_width in leaf_bit_widths]
    grid_size = _PNANOVDB_GRID_SIZE + leaf_offset + sum(leaf_sizes)
    grid_data = bytearray(grid_size)

    struct.pack_into("<Q", grid_data, 0, 0x304244566F6E614E)
    struct.pack_into("<I", grid_data, 16, 32 << 21 | 3 << 10 | 3)
    struct.pack_into("<I", grid_data, _PNANOVDB_GRID_OFF_FLAGS, _PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST)
    struct.pack_into("<IIQ", grid_data, 24, 0, 1, grid_size)
    struct.pack_into("<I", grid_data, _PNANOVDB_GRID_OFF_GRID_TYPE, _PNANOVDB_GRID_TYPE_FPN)
    struct.pack_into("<q", grid_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET, grid_size)

    tree = _PNANOVDB_GRID_SIZE
    struct.pack_into("<QQQQ", grid_data, tree, leaf_offset, lower_offset, upper_offset, root_offset)
    struct.pack_into("<III", grid_data, tree + _PNANOVDB_TREE_OFF_NODE_COUNT_LEAF, len(leaf_sizes), 1, 1)

    root = tree + root_offset
    struct.pack_into("<I", grid_data, root + _PNANOVDB_ROOT_OFF_TABLE_SIZE, 1)
    root_tile = root + _PNANOVDB_ROOT_SIZE_FPN
    struct.pack_into("<q", grid_data, root_tile + _PNANOVDB_ROOT_TILE_OFF_CHILD, upper_offset - root_offset)

    upper = tree + upper_offset
    struct.pack_into("<I", grid_data, upper + _PNANOVDB_UPPER_OFF_CHILD_MASK, 1)
    struct.pack_into("<q", grid_data, upper + _PNANOVDB_UPPER_OFF_TABLE_INDEX, lower_offset - upper_offset)

    lower = tree + lower_offset
    struct.pack_into("<I", grid_data, lower + _PNANOVDB_LOWER_OFF_CHILD_MASK, (1 << len(leaf_sizes)) - 1)
    current_leaf_offset = leaf_offset
    for index, (bit_width, leaf_size) in enumerate(zip(leaf_bit_widths, leaf_sizes, strict=True)):
        struct.pack_into(
            "<q", grid_data, lower + _PNANOVDB_LOWER_OFF_TABLE_INDEX + index * 8, current_leaf_offset - lower_offset
        )
        value_log_bits = bit_width.bit_length() - 1
        struct.pack_into("<I", grid_data, tree + current_leaf_offset + 12, value_log_bits << 29)
        current_leaf_offset += leaf_size

    return grid_data


def test_volume_rejects_unterminated_grid_name(test, device):
    """Reject grid names without a null terminator."""
    grid_data = _get_index_grid_data()
    name_offset = _PNANOVDB_GRID_OFF_GRID_NAME
    grid_data[name_offset : name_offset + _PNANOVDB_NAME_SIZE] = ord("A")

    _assert_volume_rejected(test, device, grid_data)


def test_volume_rejects_unterminated_feature_name(test, device):
    """Reject feature-array names without a null terminator."""
    grid_data = _get_index_grid_data()
    metadata_offset = struct.unpack_from("<q", grid_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET)[0]
    name_offset = metadata_offset + _PNANOVDB_GRIDBLINDMETADATA_OFF_NAME
    grid_data[name_offset : name_offset + _PNANOVDB_NAME_SIZE] = ord("A")

    _assert_volume_rejected(test, device, grid_data)


def test_volume_rejects_invalid_grid_size(test, device):
    """Reject invalid declared NanoVDB grid sizes."""
    grid_data = _get_index_grid_data()

    for case, grid_size in (("too_small", 0), ("past_buffer", grid_data.size + 1)):
        with test.subTest(case=case):
            malformed_data = grid_data.copy()
            struct.pack_into("<Q", malformed_data, _PNANOVDB_GRID_OFF_GRID_SIZE, grid_size)
            _assert_volume_rejected(test, device, malformed_data)


def test_volume_accepts_unaligned_grid_size(test, device):
    """Keep direct volume construction independent of serialized-file layout validation."""
    source = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", source, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    grid_data = bytearray(source[:grid_size])
    grid_data.append(0)
    struct.pack_into("<Q", grid_data, _PNANOVDB_GRID_OFF_GRID_SIZE, len(grid_data))

    data = wp.array(grid_data, dtype=wp.uint8, device=device)
    wp.Volume(data)


def test_volume_load_rejects_invalid_root_offset(test, device):
    """Reject serialized volumes whose tree root resolves outside the grid."""
    grid_data = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", grid_data, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    grid_data = grid_data[:grid_size].copy()
    struct.pack_into("<Q", grid_data, _PNANOVDB_GRID_SIZE + _PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT, grid_size)

    with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
        wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_rejects_invalid_node_spans(test, device):
    """Reject serialized volumes whose declared node arrays exceed the grid."""
    source = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", source, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    source = source[:grid_size]
    cases = (
        ("leaf_offset", _PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF, "<Q", grid_size),
        ("lower_offset", _PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER, "<Q", grid_size),
        ("upper_offset", _PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER, "<Q", grid_size),
        ("leaf_count", _PNANOVDB_TREE_OFF_NODE_COUNT_LEAF, "<I", 0xFFFFFFFF),
        ("lower_count", _PNANOVDB_TREE_OFF_NODE_COUNT_LOWER, "<I", 0xFFFFFFFF),
        ("upper_count", _PNANOVDB_TREE_OFF_NODE_COUNT_UPPER, "<I", 0xFFFFFFFF),
    )

    for case, field_offset, field_format, value in cases:
        with test.subTest(case=case):
            grid_data = source.copy()
            struct.pack_into(field_format, grid_data, _PNANOVDB_GRID_SIZE + field_offset, value)
            with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
                wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_rejects_invalid_root_tile_span(test, device):
    """Reject serialized volumes whose root tile table exceeds the grid."""
    grid_data = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", grid_data, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    grid_data = grid_data[:grid_size].copy()
    root_offset = struct.unpack_from("<Q", grid_data, _PNANOVDB_GRID_SIZE + _PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT)[0]
    struct.pack_into("<I", grid_data, _PNANOVDB_GRID_SIZE + root_offset + _PNANOVDB_ROOT_OFF_TABLE_SIZE, 0xFFFFFFFF)

    with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
        wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_accepts_empty_tree(test, device):
    """Accept a serialized volume with unusable offsets for empty node levels."""
    root_offset = 64
    grid_size = _PNANOVDB_GRID_SIZE + root_offset + _PNANOVDB_ROOT_SIZE_FPN
    grid_data = bytearray(grid_size)

    struct.pack_into("<Q", grid_data, 0, 0x304244566F6E614E)
    struct.pack_into("<I", grid_data, 16, 32 << 21 | 3 << 10 | 3)
    struct.pack_into("<I", grid_data, _PNANOVDB_GRID_OFF_FLAGS, _PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST)
    struct.pack_into("<IIQ", grid_data, 24, 0, 1, grid_size)
    struct.pack_into("<I", grid_data, _PNANOVDB_GRID_OFF_GRID_TYPE, _PNANOVDB_GRID_TYPE_FPN)
    struct.pack_into("<q", grid_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET, grid_size)
    struct.pack_into("<QQQQ", grid_data, _PNANOVDB_GRID_SIZE, 0, 1, (1 << 64) - 1, root_offset)

    volume = wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)
    test.assertIsNotNone(volume)


def test_volume_load_rejects_overlapping_tree_regions(test, device):
    """Reject serialized volumes whose breadth-first tree regions overlap."""
    source = _make_fpn_grid((1,))
    tree_offset = _PNANOVDB_GRID_SIZE
    leaf_offset, lower_offset, upper_offset, root_offset = struct.unpack_from("<QQQQ", source, tree_offset)
    metadata_offset = struct.unpack_from("<q", source, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET)[0]

    root_upper = source.copy()
    root = tree_offset + root_offset
    struct.pack_into("<I", root_upper, root + _PNANOVDB_ROOT_OFF_TABLE_SIZE, 2)
    struct.pack_into("<q", root_upper, tree_offset + upper_offset + _PNANOVDB_ROOT_TILE_OFF_CHILD, 0)

    upper_lower = source.copy()
    overlapping_lower_offset = lower_offset - 32
    struct.pack_into("<Q", upper_lower, tree_offset + _PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER, overlapping_lower_offset)
    struct.pack_into(
        "<q",
        upper_lower,
        tree_offset + upper_offset + _PNANOVDB_UPPER_OFF_TABLE_INDEX,
        overlapping_lower_offset - upper_offset,
    )
    lower = tree_offset + overlapping_lower_offset
    upper_lower[lower + _PNANOVDB_LOWER_OFF_CHILD_MASK : lower + _PNANOVDB_LOWER_OFF_TABLE_INDEX] = b"\0" * (
        _PNANOVDB_LOWER_OFF_TABLE_INDEX - _PNANOVDB_LOWER_OFF_CHILD_MASK
    )

    lower_leaf = source.copy()
    overlapping_leaf_offset = leaf_offset - 32
    struct.pack_into("<Q", lower_leaf, tree_offset + _PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF, overlapping_leaf_offset)
    struct.pack_into(
        "<q",
        lower_leaf,
        tree_offset + lower_offset + _PNANOVDB_LOWER_OFF_TABLE_INDEX,
        overlapping_leaf_offset - lower_offset,
    )
    struct.pack_into("<I", lower_leaf, tree_offset + overlapping_leaf_offset + 12, 0)

    leaf_metadata = source.copy()
    struct.pack_into("<q", leaf_metadata, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET, metadata_offset - 32)

    for case, grid_data in (
        ("root_upper", root_upper),
        ("upper_lower", upper_lower),
        ("lower_leaf", lower_leaf),
        ("leaf_metadata", leaf_metadata),
    ):
        with test.subTest(case=case):
            with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
                wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_rejects_invalid_child_references(test, device):
    """Reject serialized volumes whose child references escape the declared node arrays."""
    source = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", source, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    source = source[:grid_size]
    child_offsets = _find_index_grid_child_offsets(source)

    for level, child_offset in zip(("root", "upper", "lower"), child_offsets, strict=True):
        valid_reference = struct.unpack_from("<q", source, child_offset)[0]
        for case, reference in (("out_of_bounds", grid_size), ("inside_node", valid_reference + 32)):
            with test.subTest(level=level, case=case):
                grid_data = source.copy()
                struct.pack_into("<q", grid_data, child_offset, reference)
                with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
                    wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_rejects_child_references_to_empty_levels(test, device):
    """Reject child references to node levels whose declared count is zero."""
    source = _make_fpn_grid((1,))
    tree_offset = _PNANOVDB_GRID_SIZE
    cases = (
        ("root_to_upper", _PNANOVDB_TREE_OFF_NODE_COUNT_UPPER),
        ("upper_to_lower", _PNANOVDB_TREE_OFF_NODE_COUNT_LOWER),
        ("lower_to_leaf", _PNANOVDB_TREE_OFF_NODE_COUNT_LEAF),
    )

    for case, count_offset in cases:
        with test.subTest(case=case):
            grid_data = source.copy()
            struct.pack_into("<I", grid_data, tree_offset + count_offset, 0)
            with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
                wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_validates_every_grid(test, device):
    """Reject serialized multi-grid volumes when a later grid is invalid."""
    source = _get_index_grid_data()
    first_grid_size = struct.unpack_from("<Q", source, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    second_grid_size = struct.unpack_from("<Q", source, first_grid_size + _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    first_grid = source[:first_grid_size].copy()
    second_grid = source[first_grid_size : first_grid_size + second_grid_size].copy()
    struct.pack_into("<Q", second_grid, _PNANOVDB_GRID_SIZE + _PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT, second_grid_size)

    with test.subTest(case="invalid_later_grid"):
        with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
            wp.Volume.load_from_nvdb(_serialize_nvdb_grids(first_grid, second_grid), device=device)

    flags = struct.unpack_from("<I", first_grid, _PNANOVDB_GRID_OFF_FLAGS)[0]
    struct.pack_into("<I", first_grid, _PNANOVDB_GRID_OFF_FLAGS, flags & ~_PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST)
    with test.subTest(case="unsupported_then_invalid"):
        with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
            wp.Volume.load_from_nvdb(_serialize_nvdb_grids(first_grid, second_grid), device=device)


def test_volume_load_rejects_unsupported_tree_layout(test, device):
    """Report non-breadth-first NanoVDB trees as unsupported."""
    grid_data = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", grid_data, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    grid_data = grid_data[:grid_size].copy()
    flags = struct.unpack_from("<I", grid_data, _PNANOVDB_GRID_OFF_FLAGS)[0]
    struct.pack_into("<I", grid_data, _PNANOVDB_GRID_OFF_FLAGS, flags & ~_PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST)

    with test.assertRaisesRegex(RuntimeError, "Unsupported NanoVDB tree layout"):
        wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_rejects_invalid_tree_layout(test, device):
    """Reject serialized volumes with unsupported types or misaligned grid sizes."""
    source = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", source, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    source = source[:grid_size]

    with test.subTest(case="unsupported_grid_type"):
        grid_data = source.copy()
        struct.pack_into("<I", grid_data, _PNANOVDB_GRID_OFF_GRID_TYPE, 0)
        with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
            wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)

    with test.subTest(case="misaligned_grid_size"):
        grid_data = bytearray(source)
        grid_data.append(0)
        struct.pack_into("<Q", grid_data, _PNANOVDB_GRID_OFF_GRID_SIZE, len(grid_data))
        with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
            wp.Volume.load_from_nvdb(_serialize_nvdb_grid(grid_data), device=device)


def test_volume_load_accepts_fpn_leaf_spans(test, device):
    """Accept valid variable-size FpN leaf payloads."""
    valid_grid = _make_fpn_grid((1, 2))
    wp.Volume.load_from_nvdb(_serialize_nvdb_grid(valid_grid), device=device)


def test_volume_load_rejects_truncated_fpn_leaf(test, device):
    """Reject an FpN leaf whose value payload is truncated."""
    truncated_grid = _make_fpn_grid((16,))
    leaf_offset = struct.unpack_from("<Q", truncated_grid, _PNANOVDB_GRID_SIZE + _PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF)[0]
    truncated_size = _PNANOVDB_GRID_SIZE + leaf_offset + 96
    struct.pack_into("<Q", truncated_grid, _PNANOVDB_GRID_OFF_GRID_SIZE, truncated_size)
    struct.pack_into("<q", truncated_grid, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET, truncated_size)
    truncated_grid = truncated_grid[:truncated_size]
    with test.assertRaisesRegex(RuntimeError, "Invalid NanoVDB grid structure"):
        wp.Volume.load_from_nvdb(_serialize_nvdb_grid(truncated_grid), device=device)


def test_volume_load_accepts_metadata_free_grids(test, device):
    """Accept legacy NanoVDB grids that omit the unused metadata offset."""
    for path in _METADATA_FREE_GRID_PATHS:
        with test.subTest(grid=os.path.basename(path)):
            with open(path, "rb") as stream:
                volume = wp.Volume.load_from_nvdb(stream, device=device)
            test.assertEqual(volume.get_feature_array_count(), 0)


def test_volume_rejects_invalid_feature_metadata_range(test, device):
    """Reject blind metadata tables that extend beyond the grid."""
    grid_data = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", grid_data, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    metadata_offset = struct.unpack_from("<q", grid_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET)[0]

    cases = (
        ("negative_offset", -1, 1),
        ("table_past_grid", grid_size, 1),
        ("excessive_count", metadata_offset, 0xFFFFFFFF),
    )
    for case, offset, count in cases:
        with test.subTest(case=case):
            malformed_data = grid_data.copy()
            struct.pack_into("<qI", malformed_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET, offset, count)
            _assert_volume_rejected(test, device, malformed_data)


def test_volume_rejects_invalid_feature_data_range(test, device):
    """Reject feature data ranges that do not fit within the current grid.

    Cover signed-offset underflow, grid overflow, cross-grid access, and
    count-by-size multiplication overflow.
    """
    grid_data = _get_index_grid_data()
    grid_size = struct.unpack_from("<Q", grid_data, _PNANOVDB_GRID_OFF_GRID_SIZE)[0]
    metadata_offset = struct.unpack_from("<q", grid_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET)[0]
    data_offset = struct.unpack_from("<q", grid_data, metadata_offset)[0]
    data_start = metadata_offset + data_offset

    cases = (
        ("minimum_offset", -(1 << 63), 1, 1),
        ("before_grid", -metadata_offset - 1, 1, 1),
        ("past_grid", grid_size - metadata_offset + 1, 1, 1),
        ("range_past_grid", data_offset, grid_size - data_start + 1, 1),
        ("size_overflow", data_offset, 1 << 63, 3),
        ("next_grid", grid_size - metadata_offset, 1, 1),
    )
    for case, offset, count, value_size in cases:
        with test.subTest(case=case):
            malformed_data = grid_data.copy()
            struct.pack_into("<qQI", malformed_data, metadata_offset, offset, count, value_size)
            _assert_volume_rejected(test, device, malformed_data)


def test_volume_accepts_in_bounds_negative_feature_offset(test, device):
    """Accept signed feature offsets that resolve within the grid."""
    grid_data = _get_index_grid_data()
    metadata_offset = struct.unpack_from("<q", grid_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET)[0]
    struct.pack_into("<qQI", grid_data, metadata_offset, -metadata_offset, 8, 1)

    volume = wp.Volume(wp.array(grid_data, dtype=wp.uint8, device=device))
    np.testing.assert_array_equal(volume.feature_array(0, dtype=wp.uint8).numpy(), grid_data[:8])


def test_volume_snapshots_feature_metadata(test, device):
    """Verify that feature metadata remains stable after volume creation.

    Mutate the aliased source metadata after construction to confirm that
    accessors use the validated snapshot rather than the mutable buffer.
    """
    grid_data = _get_index_grid_data()
    metadata_offset = struct.unpack_from("<q", grid_data, _PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET)[0]
    data = wp.array(grid_data, dtype=wp.uint8, device=device)
    volume = wp.Volume(data, copy=False)
    expected = volume.get_feature_array_info(0)

    mutated_data = grid_data.copy()
    struct.pack_into("<qQI", mutated_data, metadata_offset, -metadata_offset - 1, 1, 1)
    name_offset = metadata_offset + _PNANOVDB_GRIDBLINDMETADATA_OFF_NAME
    mutated_data[name_offset : name_offset + _PNANOVDB_NAME_SIZE] = ord("A")
    data.assign(mutated_data)
    wp.synchronize_device(device)

    test.assertEqual(volume.get_feature_array_info(0), expected)


devices = get_test_devices()


class TestVolumeValidation(unittest.TestCase):
    def test_volume_load_rejects_invalid_root_offset_cpu(self):
        test_volume_load_rejects_invalid_root_offset(self, "cpu")

    def test_volume_load_rejects_invalid_node_spans_cpu(self):
        test_volume_load_rejects_invalid_node_spans(self, "cpu")

    def test_volume_load_rejects_invalid_root_tile_span_cpu(self):
        test_volume_load_rejects_invalid_root_tile_span(self, "cpu")

    def test_volume_load_rejects_overlapping_tree_regions_cpu(self):
        test_volume_load_rejects_overlapping_tree_regions(self, "cpu")

    def test_volume_load_rejects_invalid_child_references_cpu(self):
        test_volume_load_rejects_invalid_child_references(self, "cpu")

    def test_volume_load_rejects_child_references_to_empty_levels_cpu(self):
        test_volume_load_rejects_child_references_to_empty_levels(self, "cpu")

    def test_volume_load_validates_every_grid_cpu(self):
        test_volume_load_validates_every_grid(self, "cpu")

    def test_volume_load_rejects_unsupported_tree_layout_cpu(self):
        test_volume_load_rejects_unsupported_tree_layout(self, "cpu")

    def test_volume_load_rejects_invalid_tree_layout_cpu(self):
        test_volume_load_rejects_invalid_tree_layout(self, "cpu")

    def test_volume_load_rejects_truncated_fpn_leaf_cpu(self):
        test_volume_load_rejects_truncated_fpn_leaf(self, "cpu")

    def test_volume_rejects_missing_feature_name(self):
        """Reject feature metadata without a name pointer."""

        class Core:
            @staticmethod
            def wp_volume_get_blind_data_info(_id, _feature_index, buf, *_args):
                buf._obj.value = 1
                return None

        volume = wp.Volume.__new__(wp.Volume)
        volume.id = 1
        volume.runtime = type("Runtime", (), {"core": Core()})()

        try:
            with mock.patch.object(wp.Volume, "_decode_nvdb_name", return_value=""):
                with self.assertRaisesRegex(RuntimeError, "Invalid feature array"):
                    volume.get_feature_array_info(0)
        finally:
            volume.id = None


add_function_test(
    TestVolumeValidation,
    "test_volume_rejects_unterminated_grid_name",
    test_volume_rejects_unterminated_grid_name,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_rejects_unterminated_feature_name",
    test_volume_rejects_unterminated_feature_name,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_rejects_invalid_grid_size",
    test_volume_rejects_invalid_grid_size,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_accepts_unaligned_grid_size",
    test_volume_accepts_unaligned_grid_size,
    devices=devices,
)
add_function_test(
    TestVolumeValidation, "test_volume_load_accepts_empty_tree", test_volume_load_accepts_empty_tree, devices=devices
)
add_function_test(
    TestVolumeValidation,
    "test_volume_load_accepts_fpn_leaf_spans",
    test_volume_load_accepts_fpn_leaf_spans,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_load_accepts_metadata_free_grids",
    test_volume_load_accepts_metadata_free_grids,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_rejects_invalid_feature_metadata_range",
    test_volume_rejects_invalid_feature_metadata_range,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_rejects_invalid_feature_data_range",
    test_volume_rejects_invalid_feature_data_range,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_accepts_in_bounds_negative_feature_offset",
    test_volume_accepts_in_bounds_negative_feature_offset,
    devices=devices,
)
add_function_test(
    TestVolumeValidation,
    "test_volume_snapshots_feature_metadata",
    test_volume_snapshots_feature_metadata,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
