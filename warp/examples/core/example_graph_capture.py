# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Graph Capture
#
# Shows how to implement CUDA graph capture using Python's try-finally
# pattern. The finally block ensures wp.capture_end() gets called, even
# if an exception occurs during capture, which would otherwise
# trap the stream in a capturing state.
#
###########################################################################

import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation

wp.init()


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
        self.W = 128
        self.H = 128
        min_x, max_x = 0.0, 2.0
        min_y, max_y = 0.0, 2.0

        # create a grid of pixels
        x = np.linspace(min_x, max_x, self.W)
        y = np.linspace(min_y, max_y, self.H)

        self.x = wp.array(x, dtype=float)
        self.y = wp.array(y, dtype=float)
        self.pixel_values = wp.zeros((self.W, self.H), dtype=float)

        self.seed = 42
        self.shift = 2e-2
        self.frequency = 1.0
        self.amplitude = 1.0

        # use graph capture if launching from a CUDA-capable device
        self.use_graph = wp.get_device().is_cuda
        if self.use_graph:
            # record launches
            with wp.ScopedCapture() as capture:
                self.fbm()
            self.graph = capture.graph

    def fbm(self):
        for _ in range(16):
            wp.launch(
                kernel=fbm,
                dim=(self.H, self.W),
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
            wp.launch(kernel=slide, dim=self.W, inputs=[self.x, self.shift])

            if self.use_graph:
                wp.capture_launch(self.graph)
            else:  # cpu path
                self.fbm()

    def render(self):
        pass

    def step_and_render(self, frame_num=None, img=None):
        self.step()

        with wp.ScopedTimer("render"):
            if img:
                pixels = self.pixel_values.numpy()
                pixels = (pixels + 1.0) / 2.0
                img.set_array(pixels)

        return (img,)


if __name__ == "__main__":
    example = Example()

    # Create the animation
    fig = plt.figure()
    img = plt.imshow(example.pixel_values.numpy(), "gray", origin="lower", animated=True)
    img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))

    ani = FuncAnimation(fig, example.step_and_render, fargs=(img,), frames=1000, interval=30)

    # Display the animation
    plt.show()
