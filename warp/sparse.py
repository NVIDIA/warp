# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sparse linear algebra operations for Block Sparse Row (BSR) matrices.

This module provides GPU-accelerated sparse matrix operations for simulation, including
matrix-matrix multiplication, matrix-vector multiplication, addition, scaling, and
transpose operations. BSR format supports arbitrary block sizes, with Compressed Sparse
Row (CSR) format supported as a special case using 1x1 blocks.

The :class:`BsrMatrix` class supports operator overloading for intuitive matrix operations
(``+``, ``-``, ``*``, ``@``), and lower-level functions are available for fine-grained
control over memory allocations.

Usage:
    This module must be explicitly imported::

        import warp.sparse

See Also:
    :doc:`/domain_modules/sparse` for detailed examples and usage patterns.
"""

# isort: skip_file

from warp._src.sparse import bsr_axpy_work_arrays as bsr_axpy_work_arrays
from warp._src.sparse import bsr_mm_work_arrays as bsr_mm_work_arrays
from warp._src.sparse import BsrMatrix as BsrMatrix
from warp._src.sparse import bsr_assign as bsr_assign
from warp._src.sparse import bsr_axpy as bsr_axpy
from warp._src.sparse import bsr_block_index as bsr_block_index
from warp._src.sparse import bsr_copy as bsr_copy
from warp._src.sparse import bsr_diag as bsr_diag
from warp._src.sparse import bsr_from_triplets as bsr_from_triplets
from warp._src.sparse import bsr_get_diag as bsr_get_diag
from warp._src.sparse import bsr_identity as bsr_identity
from warp._src.sparse import bsr_matrix_t as bsr_matrix_t
from warp._src.sparse import bsr_mm as bsr_mm
from warp._src.sparse import bsr_mv as bsr_mv
from warp._src.sparse import bsr_row_index as bsr_row_index
from warp._src.sparse import bsr_scale as bsr_scale
from warp._src.sparse import bsr_set_diag as bsr_set_diag
from warp._src.sparse import bsr_set_from_triplets as bsr_set_from_triplets
from warp._src.sparse import bsr_set_identity as bsr_set_identity
from warp._src.sparse import bsr_set_transpose as bsr_set_transpose
from warp._src.sparse import bsr_set_zero as bsr_set_zero
from warp._src.sparse import bsr_transposed as bsr_transposed
from warp._src.sparse import bsr_zeros as bsr_zeros


# TODO: Remove after cleaning up the public API.

from warp._src import sparse as _sparse


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_sparse, "warp", name)
