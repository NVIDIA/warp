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

import numpy as np
import torch

import warp as wp

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

        wp.launch(kernel=eval_rosenbrock, dim=ctx.num_particles, inputs=[ctx.xy], outputs=[ctx.z])

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
            adjoint=True,
        )

        # return adjoint w.r.t. inputs
        return (wp.to_torch(ctx.xy.grad), None)


class Example:
    def __init__(self, headless=False, train_iters=10):
        self.num_particles = 1500
        self.train_iters = train_iters
        self.frame = 0

        self.learning_rate = 5e-2

        self.torch_device = wp.device_to_torch(wp.get_device())

        rng = np.random.default_rng(42)
        self.xy = torch.tensor(
            rng.normal(size=(self.num_particles, 2)), dtype=torch.float32, requires_grad=True, device=self.torch_device
        )
        self.xp_np = self.xy.numpy(force=True)
        self.opt = torch.optim.Adam([self.xy], lr=self.learning_rate)

        if headless:
            self.scatter_plot = None
            self.mean_marker = None
        else:
            self.scatter_plot = self.create_plot()

        self.mean_pos = np.empty((2,))

    def create_plot(self):
        import matplotlib.pyplot as plt

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

        # Plot optimum
        plt.plot(1, 1, "*", color="r", markersize=10)

        plt.title("Rosenbrock function")
        plt.xlabel("x")
        plt.ylabel("y")

        (self.mean_marker,) = ax.plot([], [], "o", color="w", markersize=5)

        # Create a scatter plot (initially empty)
        return ax.scatter([], [], c="k", s=2)

    def forward(self):
        self.z = Rosenbrock.apply(self.xy, self.num_particles)

    def step(self):
        self.opt.zero_grad()
        self.forward()
        self.z.backward(torch.ones_like(self.z))

        self.opt.step()

        # Update the scatter plot
        self.xy_np = self.xy.numpy(force=True)

        # Compute mean
        self.mean_pos = np.mean(self.xy_np, axis=0)
        print(f"\rFrame {self.frame:5d} particle mean: {self.mean_pos[0]:.8f}, {self.mean_pos[1]:.8f}    ", end="")

        self.frame += 1

    def render(self):
        if self.scatter_plot is None:
            return

        self.scatter_plot.set_offsets(np.c_[self.xy_np[:, 0], self.xy_np[:, 1]])
        self.mean_marker.set_data(self.mean_pos[0], self.mean_pos[1])

    # Function to update the scatter plot
    def step_and_render(self, frame):
        for _ in range(self.train_iters):
            self.step()

        self.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=10000, help="Total number of frames.")
    parser.add_argument("--train_iters", type=int, default=10, help="Total number of training iterations per frame.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(headless=args.headless, train_iters=args.train_iters)

        if not args.headless:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            # Create the animation
            ani = FuncAnimation(example.fig, example.step_and_render, frames=args.num_frames, interval=100)

            # Display the animation
            plt.show()

        else:
            for _ in range(args.num_frames):
                example.step()
