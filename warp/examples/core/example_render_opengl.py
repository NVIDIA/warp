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
# OpenGL renderer example
#
# Demonstrates how to set up tiled rendering and retrieves the pixels from
# OpenGLRenderer as a Warp array while keeping all memory on the GPU.
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


class Example:
    def __init__(self, num_tiles=4, custom_tile_arrangement=False):
        if num_tiles < 1:
            raise ValueError("num_tiles must be greater than or equal to 1.")

        self.renderer = wp.render.OpenGLRenderer(vsync=False)
        instance_ids = []

        if custom_tile_arrangement:
            positions = []
            sizes = []
        else:
            positions = None
            sizes = None

        # set up instances to hide one of the capsules in each tile
        for i in range(num_tiles):
            instances = [j for j in np.arange(13) if j != i + 2]
            instance_ids.append(instances)
            if custom_tile_arrangement:
                angle = np.pi * 2.0 / num_tiles * i
                positions.append((int(np.cos(angle) * 150 + 250), int(np.sin(angle) * 150 + 250)))
                sizes.append((150, 150))
        self.renderer.setup_tiled_rendering(instance_ids, tile_positions=positions, tile_sizes=sizes)

        self.renderer.render_ground()

    def render(self):
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)
        for i in range(10):
            self.renderer.render_capsule(
                f"capsule_{i}",
                [i - 5.0, np.sin(time + i * 0.2), -3.0],
                [0.0, 0.0, 0.0, 1.0],
                radius=0.5,
                half_height=0.8,
            )
        self.renderer.render_cylinder(
            "cylinder",
            [3.2, 1.0, np.sin(time + 0.5)],
            np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))),
            radius=0.5,
            half_height=0.8,
        )
        self.renderer.render_cone(
            "cone",
            [-1.2, 1.0, 0.0],
            np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
            radius=0.5,
            half_height=0.8,
        )
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse
    import distutils.util

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_tiles", type=int, default=4, help="Number of viewports to render in a single frame.")
    parser.add_argument(
        "--show_plot",
        type=lambda x: bool(distutils.util.strtobool(x.strip())),
        default=True,
        help="Display the pixels in an additional matplotlib figure.",
    )
    parser.add_argument("--render_mode", type=str, choices=("depth", "rgb"), default="depth", help="")
    parser.add_argument(
        "--split_up_tiles",
        type=lambda x: bool(distutils.util.strtobool(x.strip())),
        default=True,
        help="Whether to split tiles into subplots when --show_plot is True.",
    )
    parser.add_argument("--custom_tile_arrangement", action="store_true", help="Apply custom tile arrangement.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(num_tiles=args.num_tiles, custom_tile_arrangement=args.custom_tile_arrangement)

        channels = 1 if args.render_mode == "depth" else 3

        if args.show_plot:
            import matplotlib.pyplot as plt

            if args.split_up_tiles:
                pixels = wp.zeros(
                    (args.num_tiles, example.renderer.tile_height, example.renderer.tile_width, channels),
                    dtype=wp.float32,
                )
                ncols = int(np.ceil(np.sqrt(args.num_tiles)))
                nrows = int(np.ceil(args.num_tiles / float(ncols)))
                img_plots = []
                aspect_ratio = example.renderer.tile_height / example.renderer.tile_width
                fig, axes = plt.subplots(
                    ncols=ncols,
                    nrows=nrows,
                    constrained_layout=True,
                    figsize=(ncols * 3.5, nrows * 3.5 * aspect_ratio),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                    num=1,
                )
                tile_temp = np.zeros(
                    (example.renderer.tile_height, example.renderer.tile_width, channels), dtype=np.float32
                )
                for dim in range(ncols * nrows):
                    ax = axes[dim // ncols, dim % ncols]
                    if dim >= args.num_tiles:
                        ax.axis("off")
                        continue
                    if args.render_mode == "depth":
                        img_plots.append(
                            ax.imshow(
                                tile_temp,
                                vmin=example.renderer.camera_near_plane,
                                vmax=example.renderer.camera_far_plane,
                            )
                        )
                    else:
                        img_plots.append(ax.imshow(tile_temp))
            else:
                fig = plt.figure(1)
                pixels = wp.zeros(
                    (example.renderer.screen_height, example.renderer.screen_width, channels), dtype=wp.float32
                )
                if args.render_mode == "depth":
                    img_plot = plt.imshow(
                        pixels.numpy(), vmin=example.renderer.camera_near_plane, vmax=example.renderer.camera_far_plane
                    )
                else:
                    img_plot = plt.imshow(pixels.numpy())

            plt.ion()
            plt.show()

        while example.renderer.is_running():
            example.render()

            if args.show_plot and plt.fignum_exists(1):
                if args.split_up_tiles:
                    pixel_shape = (args.num_tiles, example.renderer.tile_height, example.renderer.tile_width, channels)
                else:
                    pixel_shape = (example.renderer.screen_height, example.renderer.screen_width, channels)

                if pixel_shape != pixels.shape:
                    # make sure we resize the pixels array to the right dimensions if the user resizes the window
                    pixels = wp.zeros(pixel_shape, dtype=wp.float32)

                example.renderer.get_pixels(pixels, split_up_tiles=args.split_up_tiles, mode=args.render_mode)

                if args.split_up_tiles:
                    pixels_np = pixels.numpy()
                    for i, img_plot in enumerate(img_plots):
                        img_plot.set_data(pixels_np[i])
                else:
                    img_plot.set_data(pixels.numpy())
                fig.canvas.draw()
                fig.canvas.flush_events()

        example.renderer.clear()
