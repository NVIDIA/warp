# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
import os

example_dir = os.path.dirname(os.path.realpath(__file__))

# set default cache directory before wp.init()
wp.config.kernel_cache_dir = os.path.join(example_dir, "tmp", "warpcache1")

wp.init()

print("+++ Current cache directory: ", wp.config.kernel_cache_dir)

# change cache directory after wp.init()
wp.build.init_kernel_cache(os.path.join(example_dir, "tmp", "warpcache2"))

print("+++ Current cache directory: ", wp.config.kernel_cache_dir)

# clear kernel cache (forces fresh kernel builds every time)
wp.build.clear_kernel_cache()


@wp.kernel
def basic(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = float(tid)

device = "cpu"
n = 10
x = wp.zeros(n, dtype=float, device=device)

wp.launch(kernel=basic, dim=n, inputs=[x], device=device)
print(x.numpy())
