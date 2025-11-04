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

import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


class TestGreenContextLocality(unittest.TestCase):
    @unittest.skipUnless(wp.cuda_toolkit_version_at_least(12, 4), "Green contexts require CUDA toolkit 12.4 or higher")
    def test_green_ctx(self):
        device_ordinal = 0
        contexts = wp.create_green_ctx(device_ordinal=device_ordinal, min_sms_per_partition=8)

        # Verify we got at least one context
        self.assertGreater(len(contexts), 0, "Should create at least one green context")

        for i, ctx in enumerate(contexts):
            wp.map_cuda_device(f"cuda:{device_ordinal}:{i}", int(ctx))

        # Verify each context is valid
        for i, ctx in enumerate(contexts):
            self.assertIsNotNone(ctx, f"Primary context {i} should not be None")

        n = 1024 * 1024

        streams = []
        for i in range(len(contexts)):
            alias = f"cuda:{device_ordinal}:{i}"
            s = wp.Stream(alias)
            streams.append(s)

        policy = wp.blocked()

        buffer = wp.zeros_localized((n,), dtype=float, partition_desc=policy, streams=streams)

        iters = 10

        for _ in range(iters):
            wp.launch_localized(inc, dim=n, inputs=[buffer], mapping=policy, streams=streams)

        wp.synchronize_stream(streams[0])


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
