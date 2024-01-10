# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# OpenGL renderer example
#
# Demonstrates how to set up tiled rendering and retrieves the pixels from
# OpenGLRenderer as a Warp array while keeping all memory on the GPU.
#
###########################################################################

import warp as wp
import warp.render
import numpy as np

wp.init()

# number of viewports to render in a single frame
num_tiles = 4
# whether to split tiles into subplots
split_up_tiles = True
# whether to apply custom tile arrangement
custom_tile_arrangement = False
# whether to display the pixels in a matplotlib figure
show_plot = True
# whether to render depth image to a Warp array
render_mode = "depth"

renderer = wp.render.OpenGLRenderer(vsync=False)
instance_ids = []

if custom_tile_arrangement:
    positions = []
    sizes = []
else:
    positions = None
    sizes = None

if num_tiles > 1:
    # set up instances to hide one of the capsules in each tile
    for i in range(num_tiles):
        instances = [j for j in np.arange(13) if j != i + 2]
        instance_ids.append(instances)
        if custom_tile_arrangement:
            angle = np.pi * 2.0 / num_tiles * i
            positions.append((int(np.cos(angle) * 150 + 250), int(np.sin(angle) * 150 + 250)))
            sizes.append((150, 150))
    renderer.setup_tiled_rendering(instance_ids, tile_positions=positions, tile_sizes=sizes)

renderer.render_ground()

channels = 1 if render_mode == "depth" else 3
if show_plot:
    import matplotlib.pyplot as plt

    if split_up_tiles:
        pixels = wp.zeros((num_tiles, renderer.tile_height, renderer.tile_width, channels), dtype=wp.float32)
        ncols = int(np.ceil(np.sqrt(num_tiles)))
        nrows = int(np.ceil(num_tiles / float(ncols)))
        img_plots = []
        aspect_ratio = renderer.tile_height / renderer.tile_width
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
        tile_temp = np.zeros((renderer.tile_height, renderer.tile_width, channels), dtype=np.float32)
        for dim in range(ncols * nrows):
            ax = axes[dim // ncols, dim % ncols]
            if dim >= num_tiles:
                ax.axis("off")
                continue
            if render_mode == "depth":
                img_plots.append(ax.imshow(tile_temp, vmin=renderer.camera_near_plane, vmax=renderer.camera_far_plane))
            else:
                img_plots.append(ax.imshow(tile_temp))
    else:
        fig = plt.figure(1)
        pixels = wp.zeros((renderer.screen_height, renderer.screen_width, channels), dtype=wp.float32)
        if render_mode == "depth":
            img_plot = plt.imshow(pixels.numpy(), vmin=renderer.camera_near_plane, vmax=renderer.camera_far_plane)
        else:
            img_plot = plt.imshow(pixels.numpy())

    plt.ion()
    plt.show()

while renderer.is_running():
    time = renderer.clock_time
    renderer.begin_frame(time)
    for i in range(10):
        renderer.render_capsule(
            f"capsule_{i}", [i - 5.0, np.sin(time + i * 0.2), -3.0], [0.0, 0.0, 0.0, 1.0], radius=0.5, half_height=0.8
        )
    renderer.render_cylinder(
        "cylinder",
        [3.2, 1.0, np.sin(time + 0.5)],
        np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))),
        radius=0.5,
        half_height=0.8,
    )
    renderer.render_cone(
        "cone",
        [-1.2, 1.0, 0.0],
        np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
        radius=0.5,
        half_height=0.8,
    )
    renderer.end_frame()

    if show_plot and plt.fignum_exists(1):
        if split_up_tiles:
            pixel_shape = (num_tiles, renderer.tile_height, renderer.tile_width, channels)
        else:
            pixel_shape = (renderer.screen_height, renderer.screen_width, channels)

        if pixel_shape != pixels.shape:
            # make sure we resize the pixels array to the right dimensions if the user resizes the window
            pixels = wp.zeros(pixel_shape, dtype=wp.float32)

        renderer.get_pixels(pixels, split_up_tiles=split_up_tiles, mode=render_mode)

        if split_up_tiles:
            pixels_np = pixels.numpy()
            for i, img_plot in enumerate(img_plots):
                img_plot.set_data(pixels_np[i])
        else:
            img_plot.set_data(pixels.numpy())
        fig.canvas.draw()
        fig.canvas.flush_events()

renderer.clear()
