# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""This file is used to test reloading module references."""

import warp as wp
import warp.tests.aux_test_reference as ref

wp.init()


@wp.kernel
def kern(expect: float):
    wp.expect_eq(ref.magic(), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
