# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
