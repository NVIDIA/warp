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

###########################################################################
# Example Graph Capture
#
# Shows how to implement CUDA graph capture using wp.ScopedCapture().
#
###########################################################################

import numpy as np

import warp as wp


@wp.kernel
def fbm(
    kernel_seed: int,
    frequency: float,
    amplitude: float,
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    z: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    state = wp.rand_init(kernel_seed)

    p = frequency * wp.vec2(x[j], y[i])
    n = amplitude * wp.noise(state, p)

    z[i, j] += n


@wp.kernel
def slide(x: wp.array(dtype=float), shift: float):
    tid = wp.tid()
    x[tid] += shift


class Example:
    def __init__(self):
        self.width = 128
        self.height = 128
        min_x, max_x = 0.0, 2.0
        min_y, max_y = 0.0, 2.0

        # create a grid of pixels
        x = np.linspace(min_x, max_x, self.width)
        y = np.linspace(min_y, max_y, self.height)

        self.x = wp.array(x, dtype=float)
        self.y = wp.array(y, dtype=float)
        self.pixel_values = wp.zeros((self.width, self.height), dtype=float)

        self.seed = 42
        self.shift = 2e-2
        self.frequency = 1.0
        self.amplitude = 1.0

        # use graph capture if launching from a CUDA-capable device
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            # record launches
            with wp.ScopedCapture() as capture:
                self.fbm()
            self.graph = capture.graph

    def fbm(self):
        for _ in range(16):
            wp.launch(
                kernel=fbm,
                dim=(self.height, self.width),
                inputs=[self.seed, self.frequency, self.amplitude, self.x, self.y],
                outputs=[self.pixel_values],
            )
            self.frequency *= 2.0
            self.amplitude *= 0.5

    def step(self):
        self.pixel_values.zero_()
        self.frequency = 1.0
        self.amplitude = 1.0

        with wp.ScopedTimer("step", active=True):
            wp.launch(kernel=slide, dim=self.width, inputs=[self.x, self.shift])

            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:  # cpu path
                self.fbm()

    def step_and_render(self, frame_num=None, img=None):
        self.step()

        with wp.ScopedTimer("render"):
            if img:
                pixels = self.pixel_values.numpy()
                pixels = (pixels + 1.0) / 2.0
                img.set_array(pixels)

        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        if not args.headless:
            import matplotlib.colors
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            # Create the animation
            fig = plt.figure()
            img = plt.imshow(example.pixel_values.numpy(), "gray", origin="lower", animated=True)
            img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))

            ani = FuncAnimation(fig, example.step_and_render, fargs=(img,), frames=1000, interval=30)

            # Display the animation
            plt.show()

        else:
            for _ in range(args.num_frames):
                example.step()
