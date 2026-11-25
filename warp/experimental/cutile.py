# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental interop for calling Warp functions from kernels written in cuTile Python.

The cuTile interop API is experimental and subject to change without a formal
deprecation cycle; it builds on cuTile preview interfaces that may change
between CUDA Toolkit and cuTile Python releases. See :doc:`the cuTile
interoperability guide </user_guide/interoperability/cutile>` for the full
contract.
"""

from warp._src.cutile import array_pointer_args as array_pointer_args
from warp._src.cutile import encode_struct as encode_struct
from warp._src.cutile import export_func as export_func

__all__ = [
    "array_pointer_args",
    "encode_struct",
    "export_func",
]
