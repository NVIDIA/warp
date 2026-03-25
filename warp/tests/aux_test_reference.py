# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file is used to test reloading module references.

import warp as wp
import warp.tests.aux_test_reference_reference as refref


@wp.func
def magic():
    return 2.0 * refref.more_magic()
