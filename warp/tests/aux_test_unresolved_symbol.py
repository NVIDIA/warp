# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.kernel
def unresolved_symbol_kernel():
    # this should trigger an exception due to unresolved symbol
    x = missing_symbol  # noqa: F821
