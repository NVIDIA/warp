# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp
from warp import stf_experimental as wp_stf


@unittest.skipUnless(wp.is_cuda_available() and wp_stf.is_available(), "CUDASTF is not available")
class TestSTFExperimental(unittest.TestCase):
    def test_smoke_stream_context(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            with wp_stf.context(stream=wp.get_stream(device)) as ctx:
                token = ctx.token()
                with wp_stf.task(ctx, token.write()):
                    pass


if __name__ == "__main__":
    unittest.main()
