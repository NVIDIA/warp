# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Dummy class used in test_reload.py"""

import warp as wp


class ClassKernelTest:
    def __init__(self, device):
        # 3x3 frames in the rest pose:
        self.identities = wp.zeros(shape=10, dtype=wp.mat33, device=device)
        wp.launch(kernel=self.gen_identities_kernel, dim=10, inputs=[self.identities], device=device)

    @wp.func
    def return_identity(e: int):
        return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    @wp.kernel
    def gen_identities_kernel(s: wp.array(dtype=wp.mat33)):
        tid = wp.tid()
        s[tid] = ClassKernelTest.return_identity(tid)
