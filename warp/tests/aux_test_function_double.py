# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Function target fixture imported as a Python module."""

import warp as wp


@wp.func
def function_external_module_double_it(x: float):
    return x * 2.0
