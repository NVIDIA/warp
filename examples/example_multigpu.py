# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Multi-GPU
#
# A basic example that shows how to allocate arrays and launch kernels
# on all available CUDA devices.
#
###########################################################################

import warp as wp

wp.init()


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


# get all CUDA devices
devices = wp.get_cuda_devices()
device_count = len(devices)

# number of launches
iters = 1000

# list of arrays, one per device
arrs = []

# loop over all devices
for device in devices:

    # use a ScopedDevice to set the target device
    with wp.ScopedDevice(device):

        # allocate array
        a = wp.zeros(250 * 1024 * 1024, dtype=float)
        arrs.append(a)

        # launch kernels
        for _ in range(iters):
            wp.launch(inc, dim=a.size, inputs=[a])

# synchronize all devices
wp.synchronize()

# print results
for i in range(device_count):
    print(f"{arrs[i].device} -> {arrs[i].numpy()}")
