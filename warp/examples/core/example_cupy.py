# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Example CuPy
#
# The example demonstrates interoperability with CuPy on CUDA devices
# and NumPy on CPU devices.
###########################################################################

import warp as wp


@wp.kernel
def saxpy(x: wp.array(dtype=float), y: wp.array(dtype=float), a: float):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


class Example:
    def __init__(self):
        device = wp.get_device()

        self.n = 10
        self.a = 1.0

        if device.is_cuda:
            # use CuPy arrays on CUDA devices
            import cupy as cp

            print(f"Using CuPy on device {device}")

            # tell CuPy to use the same device
            with cp.cuda.Device(device.ordinal):
                self.x = cp.arange(self.n, dtype=cp.float32)
                self.y = cp.ones(self.n, dtype=cp.float32)
        else:
            # use NumPy arrays on CPU
            import numpy as np

            print("Using NumPy on CPU")

            self.x = np.arange(self.n, dtype=np.float32)
            self.y = np.ones(self.n, dtype=np.float32)

    def step(self):
        # Launch a Warp kernel on the pre-allocated arrays.
        # When running on a CUDA device, these are CuPy arrays.
        # When running on the CPU, these are NumPy arrays.
        #
        # Note that the arrays can be passed to Warp kernels directly.  Under the hood,
        # Warp uses the __cuda_array_interface__ and __array_interface__ protocols to
        # access the data.
        wp.launch(saxpy, dim=self.n, inputs=[self.x, self.y, self.a])

    def render(self):
        print(self.y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=10, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        for _ in range(args.num_frames):
            example.step()
            example.render()
