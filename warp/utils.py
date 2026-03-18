# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities supporting Warp's high-level workflows."""

# isort: skip_file

# category: Array Operations

from warp._src.utils import array_cast as array_cast
from warp._src.utils import array_inner as array_inner
from warp._src.utils import array_scan as array_scan
from warp._src.utils import array_sum as array_sum


# category: Sorting

from warp._src.utils import radix_sort_pairs as radix_sort_pairs
from warp._src.utils import runlength_encode as runlength_encode
from warp._src.utils import segmented_sort_pairs as segmented_sort_pairs


# category: Graph Coloring

from warp._src.coloring import GraphColoringAlgorithm as GraphColoringAlgorithm
from warp._src.coloring import graph_coloring_assign as graph_coloring_assign
from warp._src.coloring import graph_coloring_balance as graph_coloring_balance
from warp._src.coloring import graph_coloring_get_groups as graph_coloring_get_groups


# category: Misc

from warp._src.utils import create_warp_function as create_warp_function


# TODO: Remove after cleaning up the public API.

from warp._src import utils as _utils


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_utils, "warp", name)
