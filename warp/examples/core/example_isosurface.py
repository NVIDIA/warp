# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Isosurface
#
# Shows how to use the built-in isosurface extraction functionality to
# extract a mesh from a density field. The --method option switches
# between the available wp.geometry.IsoSurfaceBase backends so their outputs can
# be compared on the same scene, and --topology selects the type of the
# extracted faces for the backends offering more than one.
#
# Note: requires a CUDA-capable device
###########################################################################


import numpy as np

import warp as wp
import warp.geometry
import warp.render

METHODS = {
    "marching_cubes": wp.geometry.IsoSurfaceMarchingCubes,
    "surface_nets": wp.geometry.IsoSurfaceNets,
}


@wp.func
def sdf_create_box(pos: wp.vec3, size: wp.vec3):
    """Create a SDF box primitive."""
    # https://iquilezles.org/articles/distfunctions
    q = wp.vec3(
        wp.abs(pos[0]) - size[0],
        wp.abs(pos[1]) - size[1],
        wp.abs(pos[2]) - size[2],
    )
    qp = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    return wp.length(qp) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)


@wp.func
def sdf_create_torus(pos: wp.vec3, major_radius: float, minor_radius: float):
    """Create a SDF torus primitive."""
    # https://iquilezles.org/articles/distfunctions
    q = wp.vec2(wp.length(wp.vec2(pos[0], pos[2])) - major_radius, pos[1])
    return wp.length(q) - minor_radius


@wp.func
def sdf_translate(pos: wp.vec3, offset: wp.vec3):
    """Translate a SDF position vector with an offset."""
    return pos - offset


@wp.func
def sdf_rotate(pos: wp.vec3, angles: wp.vec3):
    """Rotate a SDF position vector using Euler angles."""
    rot = wp.quat_rpy(
        wp.radians(angles[0]),
        wp.radians(angles[1]),
        wp.radians(angles[2]),
    )
    return wp.quat_rotate_inv(rot, pos)


@wp.func
def sdf_smooth_min(a: float, b: float, radius: float):
    """Create a SDF torus primitive."""
    # https://iquilezles.org/articles/smin
    h = wp.max(radius - wp.abs(a - b), 0.0) / radius
    return wp.min(a, b) - h * h * h * radius * (1.0 / 6.0)


@wp.kernel(enable_backward=False)
def make_field(
    torus_altitude: float,
    torus_major_radius: float,
    torus_minor_radius: float,
    smooth_min_radius: float,
    dim: int,
    time: float,
    out_data: wp.array3d[float],
):
    """Kernel to generate a SDF volume based on primitives."""
    i, j, k = wp.tid()

    # Retrieve the position of the current cell in a normalized [-1, 1] range
    # for each dimension.
    pos = wp.vec3(
        2.0 * ((float(i) + 0.5) / float(dim)) - 1.0,
        2.0 * ((float(j) + 0.5) / float(dim)) - 1.0,
        2.0 * ((float(k) + 0.5) / float(dim)) - 1.0,
    )

    box = sdf_create_box(
        sdf_translate(pos, wp.vec3(0.0, -0.7, 0.0)),
        wp.vec3(0.9, 0.3, 0.9),
    )
    torus = sdf_create_torus(
        sdf_rotate(
            sdf_translate(pos, wp.vec3(0.0, torus_altitude, 0.0)),
            wp.vec3(wp.sin(time) * 90.0, wp.cos(time) * 45.0, 0.0),
        ),
        torus_major_radius,
        torus_minor_radius,
    )
    out_data[i, j, k] = sdf_smooth_min(box, torus, smooth_min_radius)


class Example:
    def __init__(
        self, stage_path="example_isosurface.usd", verbose=False, method="marching_cubes", topology="triangle"
    ):
        self.verbose = verbose
        self.topology = topology

        self.dim = 64

        self.torus_altitude = -0.5
        self.torus_major_radius = 0.5
        self.torus_minor_radius = 0.1
        self.smooth_min_radius = 0.5

        self.fps = 60
        self.frame = 0

        self.field = wp.zeros((self.dim, self.dim, self.dim), dtype=float)

        # The rest of the example only relies on the wp.geometry.IsoSurfaceBase
        # interface, so extraction methods can be swapped freely.
        extractor_class = METHODS[method]
        options = {}
        if extractor_class is wp.geometry.IsoSurfaceNets:
            options["topology"] = topology
        elif topology != "triangle":
            raise ValueError(f"The '{method}' method only supports the 'triangle' topology.")

        self.iso = extractor_class(self.dim, self.dim, self.dim, **options)

        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

    def step(self):
        with wp.ScopedTimer("step"):
            with wp.ScopedTimer("Update Field", active=self.verbose):
                wp.launch(
                    make_field,
                    dim=self.field.shape,
                    inputs=(
                        self.torus_altitude,
                        self.torus_major_radius,
                        self.torus_minor_radius,
                        self.smooth_min_radius,
                        self.dim,
                        self.frame / self.fps,
                    ),
                    outputs=(self.field,),
                )

            with wp.ScopedTimer("Surface Extraction", active=self.verbose):
                self.iso.surface(self.field, 0.0)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("Render"):
            indices = self.iso.indices.numpy()
            if self.topology == "quad":
                # The renderer only draws triangle meshes, so split each
                # extracted quad along its first diagonal.
                quads = indices.reshape(-1, 4)
                indices = np.hstack((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)])).reshape(-1)

            self.renderer.begin_frame(self.frame / self.fps)
            self.renderer.render_mesh(
                "surface",
                self.iso.verts.numpy(),
                indices,
                colors=(0.35, 0.55, 0.9),
                update_topology=True,
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_isosurface.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=240, help="Total number of frames.")
    parser.add_argument(
        "--method",
        type=str,
        choices=tuple(METHODS.keys()),
        default="marching_cubes",
        help="Isosurface extraction method to use.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=("triangle", "quad"),
        default="triangle",
        help="Type of the extracted faces, where anything but triangles requires the 'surface_nets' method.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose, method=args.method, topology=args.topology)
        for _ in range(args.num_frames):
            example.step()
            example.render()
            example.frame += 1

        if example.renderer is not None:
            example.renderer.save()
