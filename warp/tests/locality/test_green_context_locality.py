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

import cuda.bindings.driver as cu

import warp as wp
from warp.tests.unittest_utils import *


def cuda_toolkit_version_at_least(major, minor):
    """Check if CUDA toolkit version is at least the specified version."""
    if not wp.is_cuda_available():
        return False
    toolkit_version = wp._src.context.runtime.toolkit_version
    if toolkit_version is None:
        return False
    return toolkit_version >= (major, minor)


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


def create_green_ctx(device_ordinal=0, min_sms_per_partition=8):
    """Create green contexts for each SM partition on the device.

    Args:
        device_ordinal: CUDA device ordinal
        min_sms_per_partition: Minimum number of SMs per partition

    Returns:
        list: List of primary contexts (CUcontext) for each partition
    """
    # 0) init + pick device
    cu.cuInit(0)
    err, dev = cu.cuDeviceGet(device_ordinal)
    assert err == cu.CUresult.CUDA_SUCCESS, f"cuDeviceGet failed: {err}"

    # 1) query device SM resource
    err, sm_res = cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    assert err == cu.CUresult.CUDA_SUCCESS, f"cuDeviceGetDevResource failed: {err}"

    # 2) split SMs into groups of at least 'min_sms_per_partition'
    # first call to learn number of groups (query mode)
    err, result, nb_groups, remaining = cu.cuDevSmResourceSplitByCount(
        nbGroups=0,  # query mode
        input_=sm_res,
        useFlags=0,
        minCount=min_sms_per_partition,
    )
    assert err == cu.CUresult.CUDA_SUCCESS and nb_groups > 0, "split (count) failed"

    # second call to actually split with the correct number of groups
    err, groups, nb_groups_actual, remaining = cu.cuDevSmResourceSplitByCount(
        nbGroups=nb_groups, input_=sm_res, useFlags=0, minCount=min_sms_per_partition
    )
    assert err == cu.CUresult.CUDA_SUCCESS, "split (fill) failed"

    # 3-5) Create a green context for each partition group
    contexts = []
    flags = cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM

    for i, group in enumerate(groups):
        # Build a resource descriptor for this partition
        err, desc = cu.cuDevResourceGenerateDesc(resources=[group], nbResources=1)
        assert err == cu.CUresult.CUDA_SUCCESS, f"cuDevResourceGenerateDesc failed for group {i}"

        # Create the green context
        err, green = cu.cuGreenCtxCreate(desc, dev, flags)
        assert err == cu.CUresult.CUDA_SUCCESS, f"cuGreenCtxCreate failed for group {i}: {err}"

        # Convert to a primary CUcontext
        err, ctx = cu.cuCtxFromGreenCtx(green)
        assert err == cu.CUresult.CUDA_SUCCESS, f"cuCtxFromGreenCtx failed for group {i}: {err}"

        contexts.append(ctx)

    return contexts


class TestGreenContextLocality(unittest.TestCase):
    @unittest.skipUnless(cuda_toolkit_version_at_least(12, 4), "Green contexts require CUDA toolkit 12.4 or higher")
    def test_green_ctx(self):
        device_ordinal = 0
        contexts = create_green_ctx(device_ordinal=device_ordinal, min_sms_per_partition=8)

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
