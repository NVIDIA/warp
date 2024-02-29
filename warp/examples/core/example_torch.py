# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Torch
#
# Optimizes the Rosenbrock function using the PyTorch Adam optimizer
# The Rosenbrock function is a non-convex function, and is often used
# to test optimization algorithms. The function is defined as:
#              f(x, y) = (a - x)^2 + b * (y - x^2)^2
# where a = 1 and b = 100. The minimum value of the function is 0 at (1, 1).
#
# The example demonstrates how to set up a torch.autograd.Function to 
# incorporate Warp kernel launches within a PyTorch graph.
###########################################################################

import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

wp.init()


pvec2 = wp.types.vector(length=2, dtype=wp.float32)

# Define the Rosenbrock function
@wp.func
def rosenbrock(x: float, y: float):
    return (1.0 - x) ** 2.0 + 100.0 * (y - x**2.0) ** 2.0

@wp.kernel
def eval_rosenbrock(
    xs: wp.array(dtype=pvec2),
    # outputs
    z: wp.array(dtype=float),
):
    i = wp.tid()
    x = xs[i]
    z[i] = rosenbrock(x[0], x[1])


class Rosenbrock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, num_particles):

        ctx.xy = wp.from_torch(xy, dtype=pvec2, requires_grad=True)
        ctx.num_particles = num_particles

        # allocate output
        ctx.z = wp.zeros(num_particles, requires_grad=True)

        wp.launch(
            kernel=eval_rosenbrock,
            dim=ctx.num_particles,
            inputs=[ctx.xy],
            outputs=[ctx.z]
        )

        return wp.to_torch(ctx.z)

    @staticmethod
    def backward(ctx, adj_z):

        # map incoming Torch grads to our output variables
        ctx.z.grad = wp.from_torch(adj_z)

        wp.launch(
            kernel=eval_rosenbrock,
            dim=ctx.num_particles,
            inputs=[ctx.xy],
            outputs=[ctx.z],
            adj_inputs=[ctx.xy.grad],
            adj_outputs=[ctx.z.grad],
            adjoint=True
        )

        # return adjoint w.r.t. inputs
        return (wp.to_torch(ctx.xy.grad), None)


class Example:

    def __init__(self):

        self.num_particles = 1500

        min_x, max_x = -2.0, 2.0
        min_y, max_y = -2.0, 2.0

        # Create a grid of points
        x = np.linspace(min_x, max_x, 100)
        y = np.linspace(min_y, max_y, 100)
        X, Y = np.meshgrid(x, y)
        xy = np.column_stack((X.flatten(), Y.flatten()))
        N = len(xy)
        
        xy = wp.array(xy, dtype=pvec2)
        Z = wp.empty(N, dtype=wp.float32)

        wp.launch(eval_rosenbrock, dim=N, inputs=[xy], outputs=[Z])
        Z = Z.numpy().reshape(X.shape)

        # Plot the function as a heatmap
        self.fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        plt.imshow(
            Z,
            extent=[min_x, max_x, min_y, max_y],
            origin="lower",
            interpolation="bicubic",
            cmap="coolwarm",
        )
        plt.contour(
            X,
            Y,
            Z,
            extent=[min_x, max_x, min_y, max_y],
            levels=150,
            colors="k",
            alpha=0.5,
            linewidths=0.5,
        )

        plt.title("Rosenbrock function")
        plt.xlabel("x")
        plt.ylabel("y")

        # Create a scatter plot (initially empty)
        self.scat = ax.scatter([], [], c="k", s=2)

        # Plot optimum
        plt.plot(1, 1, "*", color="r", markersize=10)

        self.learning_rate = 5e-2

        self.torch_device = wp.device_to_torch(wp.get_device())

        rng = np.random.default_rng(42)
        self.xy = torch.tensor(rng.normal(size=(self.num_particles, 2)), dtype=torch.float32, requires_grad=True, device=self.torch_device)
        self.opt = torch.optim.Adam([self.xy], lr=self.learning_rate)

    def forward(self):
        self.z = Rosenbrock.apply(self.xy, self.num_particles)

    def step(self):
        self.opt.zero_grad()
        self.forward()
        self.z.backward(torch.ones_like(self.z))

        self.opt.step()

    def render(self):
        # Update the scatter plot
        xy_np = self.xy.numpy(force=True)
        self.scat.set_offsets(np.c_[xy_np[:, 0], xy_np[:, 1]])

        print(f"\rParticle mean: {np.mean(xy_np, axis=0)}     ", end="")

    # Function to update the scatter plot
    def step_and_render(self, frame):
        for _ in range(10):
            self.step()

        self.render()


if __name__ == "__main__":

    example = Example()

    # Create the animation
    ani = FuncAnimation(example.fig, example.step_and_render, frames=10000, interval=200)

    # Display the animation
    plt.show()
