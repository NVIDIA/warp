# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dummy module used in test_reload.py"""

import warp as wp


@wp.kernel
def k():
    pass
