# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import warp as wp


@wp.kernel
def arange(out: wp.array(dtype=int)):
    tid = wp.tid()
    out[tid] = tid


device = "cuda:0"
cmds = []

n = 10
arrays = []

for _i in range(5):
    arrays.append(wp.zeros(n, dtype=int, device=device))

# setup CUDA graph
wp.capture_begin()

# launch kernels and keep command object around
for i in range(5):
    cmd = wp.launch(arange, dim=n, inputs=[arrays[i]], device=device, record_cmd=True)
    cmds.append(cmd)

graph = wp.capture_end()

# ---------------------------------------

ref = np.arange(0, n, dtype=int)
wp.capture_launch(graph)

for i in range(5):
    print(arrays[i].numpy())


# ---------------------------------------

n = 16
arrays = []

for _i in range(5):
    arrays.append(wp.zeros(n, dtype=int, device=device))

# update graph params
for i in range(5):
    cmd.set_dim(n)
    cmd.set_param(arrays[i])

    cmd.update_graph()


wp.capture_launch(graph)
wp.synchronize()

ref = np.arange(0, n, dtype=int)

for i in range(5):
    print(arrays[i].numpy())
