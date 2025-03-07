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

from collections import defaultdict

import numpy as np

import warp as wp
import warp.render
import warp.sim
from warp.render.utils import solidify_mesh, tab10_color_map

# TODO allow NaNs in Warp kernels
NAN = wp.constant(-1.0e8)


@wp.kernel
def compute_contact_points(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    # outputs
    contact_pos0: wp.array(dtype=wp.vec3),
    contact_pos1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        contact_pos0[tid] = wp.vec3(NAN, NAN, NAN)
        contact_pos1[tid] = wp.vec3(NAN, NAN, NAN)
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        contact_pos0[tid] = wp.vec3(NAN, NAN, NAN)
        contact_pos1[tid] = wp.vec3(NAN, NAN, NAN)
        return

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    contact_pos0[tid] = wp.transform_point(X_wb_a, contact_point0[tid])
    contact_pos1[tid] = wp.transform_point(X_wb_b, contact_point1[tid])


def CreateSimRenderer(renderer):
    class SimRenderer(renderer):
        use_unique_colors = True

        def __init__(
            self,
            model: warp.sim.Model,
            path,
            scaling=1.0,
            fps=60,
            up_axis=None,
            show_rigid_contact_points=False,
            contact_points_radius=1e-3,
            show_joints=False,
            **render_kwargs,
        ):
            # create USD stage
            if up_axis is None:
                up_axis = "XYZ"[model.up_axis]
            super().__init__(path, scaling=scaling, fps=fps, up_axis=up_axis, **render_kwargs)
            self.scaling = scaling
            self.cam_axis = "XYZ".index(up_axis.upper())
            self.show_rigid_contact_points = show_rigid_contact_points
            self.show_joints = show_joints
            self.contact_points_radius = contact_points_radius
            self.populate(model)

        def populate(self, model: warp.sim.Model):
            self.skip_rendering = False

            self.model = model
            self.num_envs = model.num_envs
            self.body_names = []

            if self.show_rigid_contact_points and model.rigid_contact_max:
                self.contact_points0 = wp.array(
                    np.zeros((model.rigid_contact_max, 3)), dtype=wp.vec3, device=model.device
                )
                self.contact_points1 = wp.array(
                    np.zeros((model.rigid_contact_max, 3)), dtype=wp.vec3, device=model.device
                )

                self.contact_points0_colors = [(1.0, 0.5, 0.0)] * model.rigid_contact_max
                self.contact_points1_colors = [(0.0, 0.5, 1.0)] * model.rigid_contact_max

            self.body_env = []  # mapping from body index to its environment index
            env_id = 0
            self.bodies_per_env = model.body_count // self.num_envs
            # create rigid body nodes
            for b in range(model.body_count):
                body_name = f"body_{b}_{self.model.body_name[b].replace(' ', '_')}"
                self.body_names.append(body_name)
                self.register_body(body_name)
                if b > 0 and b % self.bodies_per_env == 0:
                    env_id += 1
                self.body_env.append(env_id)

            # create rigid shape children
            if self.model.shape_count:
                # mapping from hash of geometry to shape ID
                self.geo_shape = {}

                self.instance_count = 0

                self.body_name = {}  # mapping from body name to its body ID
                self.body_shapes = defaultdict(list)  # mapping from body index to its shape IDs

                shape_body = model.shape_body.numpy()
                shape_geo_src = model.shape_geo_src
                shape_geo_type = model.shape_geo.type.numpy()
                shape_geo_scale = model.shape_geo.scale.numpy()
                shape_geo_thickness = model.shape_geo.thickness.numpy()
                shape_geo_is_solid = model.shape_geo.is_solid.numpy()
                shape_transform = model.shape_transform.numpy()
                shape_visible = model.shape_visible.numpy()

                p = np.zeros(3, dtype=np.float32)
                q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                scale = np.ones(3)
                color = (1.0, 1.0, 1.0)
                # loop over shapes excluding the ground plane
                for s in range(model.shape_count - 1):
                    geo_type = shape_geo_type[s]
                    geo_scale = [float(v) for v in shape_geo_scale[s]]
                    geo_thickness = float(shape_geo_thickness[s])
                    geo_is_solid = bool(shape_geo_is_solid[s])
                    geo_src = shape_geo_src[s]
                    name = f"shape_{s}"

                    # shape transform in body frame
                    body = int(shape_body[s])
                    if body >= 0 and body < len(self.body_names):
                        body = self.body_names[body]
                    else:
                        body = None

                    if self.use_unique_colors and body is not None:
                        color = self._get_new_color()

                    # shape transform in body frame
                    X_bs = wp.transform_expand(shape_transform[s])
                    # check whether we can instance an already created shape with the same geometry
                    geo_hash = hash((int(geo_type), geo_src, *geo_scale, geo_thickness, geo_is_solid))
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        if geo_type == warp.sim.GEO_PLANE:
                            if s == model.shape_count - 1 and not model.ground:
                                continue  # hide ground plane

                            # plane mesh
                            width = geo_scale[0] if geo_scale[0] > 0.0 else 100.0
                            length = geo_scale[1] if geo_scale[1] > 0.0 else 100.0

                            shape = self.render_plane(
                                name, p, q, width, length, color, parent_body=body, is_template=True
                            )

                        elif geo_type == warp.sim.GEO_SPHERE:
                            shape = self.render_sphere(
                                name, p, q, geo_scale[0], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == warp.sim.GEO_CAPSULE:
                            shape = self.render_capsule(
                                name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == warp.sim.GEO_CYLINDER:
                            shape = self.render_cylinder(
                                name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == warp.sim.GEO_CONE:
                            shape = self.render_cone(
                                name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == warp.sim.GEO_BOX:
                            shape = self.render_box(
                                name, p, q, geo_scale, parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == warp.sim.GEO_MESH:
                            if not geo_is_solid:
                                faces, vertices = solidify_mesh(geo_src.indices, geo_src.vertices, geo_thickness)
                            else:
                                faces, vertices = geo_src.indices, geo_src.vertices

                            shape = self.render_mesh(
                                name,
                                vertices,
                                faces,
                                pos=p,
                                rot=q,
                                scale=geo_scale,
                                colors=color,
                                parent_body=body,
                                is_template=True,
                            )

                        elif geo_type == warp.sim.GEO_SDF:
                            continue

                        self.geo_shape[geo_hash] = shape

                    if shape_visible[s]:
                        # TODO support dynamic visibility
                        self.add_shape_instance(
                            name, shape, body, X_bs.p, X_bs.q, scale, custom_index=s, visible=shape_visible[s]
                        )
                    self.instance_count += 1

                if self.show_joints and model.joint_count:
                    joint_type = model.joint_type.numpy()
                    joint_axis = model.joint_axis.numpy()
                    joint_axis_start = model.joint_axis_start.numpy()
                    joint_axis_dim = model.joint_axis_dim.numpy()
                    joint_parent = model.joint_parent.numpy()
                    joint_child = model.joint_child.numpy()
                    joint_tf = model.joint_X_p.numpy()
                    shape_collision_radius = model.shape_collision_radius.numpy()
                    y_axis = wp.vec3(0.0, 1.0, 0.0)
                    color = (1.0, 0.0, 1.0)

                    shape = self.render_arrow(
                        "joint_arrow",
                        None,
                        None,
                        base_radius=0.01,
                        base_height=0.4,
                        cap_radius=0.02,
                        cap_height=0.1,
                        parent_body=None,
                        is_template=True,
                        color=color,
                    )
                    for i, t in enumerate(joint_type):
                        if t not in {
                            warp.sim.JOINT_REVOLUTE,
                            # warp.sim.JOINT_PRISMATIC,
                            warp.sim.JOINT_UNIVERSAL,
                            warp.sim.JOINT_COMPOUND,
                            warp.sim.JOINT_D6,
                        }:
                            continue
                        tf = joint_tf[i]
                        body = int(joint_parent[i])
                        # if body == -1:
                        #     continue
                        num_linear_axes = int(joint_axis_dim[i][0])
                        num_angular_axes = int(joint_axis_dim[i][1])

                        # find a good scale for the arrow based on the average radius
                        # of the shapes attached to the joint child body
                        scale = np.ones(3)
                        child = int(joint_child[i])
                        if child >= 0:
                            radii = []
                            for s in model.body_shapes[child]:
                                radii.append(shape_collision_radius[s])
                            if len(radii) > 0:
                                scale *= np.mean(radii) * 2.0

                        for a in range(num_linear_axes, num_linear_axes + num_angular_axes):
                            index = joint_axis_start[i] + a
                            axis = joint_axis[index]
                            if np.linalg.norm(axis) < 1e-6:
                                continue
                            p = wp.vec3(tf[:3])
                            q = wp.quat(tf[3:])
                            # compute rotation between axis and y
                            axis = axis / np.linalg.norm(axis)
                            q = q * wp.quat_between_vectors(wp.vec3(axis), y_axis)
                            name = f"joint_{i}_{a}"
                            self.add_shape_instance(name, shape, body, p, q, scale, color1=color, color2=color)
                            self.instance_count += 1

            if model.ground:
                self.render_ground(plane=model.ground_plane_params)

            if hasattr(self, "complete_setup"):
                self.complete_setup()

        def _get_new_color(self):
            return tab10_color_map(self.instance_count)

        def render(self, state: warp.sim.State):
            """
            Updates the renderer with the given simulation state.

            Args:
                state (warp.sim.State): The simulation state to render.
            """
            if self.skip_rendering:
                return

            if self.model.particle_count:
                particle_q = state.particle_q.numpy()

                # render particles
                self.render_points(
                    "particles", particle_q, radius=self.model.particle_radius.numpy(), colors=(0.8, 0.3, 0.2)
                )

                # render tris
                if self.model.tri_count:
                    self.render_mesh(
                        "surface",
                        particle_q,
                        self.model.tri_indices.numpy().flatten(),
                        colors=(0.75, 0.25, 0.0),
                    )

                # render springs
                if self.model.spring_count:
                    self.render_line_list(
                        "springs", particle_q, self.model.spring_indices.numpy().flatten(), (0.25, 0.5, 0.25), 0.02
                    )

            # render muscles
            if self.model.muscle_count:
                body_q = state.body_q.numpy()

                muscle_start = self.model.muscle_start.numpy()
                muscle_links = self.model.muscle_bodies.numpy()
                muscle_points = self.model.muscle_points.numpy()
                muscle_activation = self.model.muscle_activation.numpy()

                # for s in self.skeletons:

                #     # for mesh, link in s.mesh_map.items():

                #     #     if link != -1:
                #     #         X_sc = wp.transform_expand(self.state.body_X_sc[link].tolist())

                #     #         #self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)
                #     #         self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)

                for m in range(self.model.muscle_count):
                    start = int(muscle_start[m])
                    end = int(muscle_start[m + 1])

                    points = []

                    for w in range(start, end):
                        link = muscle_links[w]
                        point = muscle_points[w]

                        X_sc = wp.transform_expand(body_q[link][0])

                        points.append(wp.transform_point(X_sc, point).tolist())

                    self.render_line_strip(
                        name=f"muscle_{m}", vertices=points, radius=0.0075, color=(muscle_activation[m], 0.2, 0.5)
                    )

            # update bodies
            if self.model.body_count:
                self.update_body_transforms(state.body_q)

                if self.show_rigid_contact_points and self.model.rigid_contact_max:
                    wp.launch(
                        kernel=compute_contact_points,
                        dim=self.model.rigid_contact_max,
                        inputs=[
                            state.body_q,
                            self.model.shape_body,
                            self.model.rigid_contact_count,
                            self.model.rigid_contact_shape0,
                            self.model.rigid_contact_shape1,
                            self.model.rigid_contact_point0,
                            self.model.rigid_contact_point1,
                        ],
                        outputs=[
                            self.contact_points0,
                            self.contact_points1,
                        ],
                        device=self.model.device,
                    )

                    self.render_points(
                        "contact_points0",
                        self.contact_points0.numpy(),
                        radius=self.contact_points_radius * self.scaling,
                        colors=self.contact_points0_colors,
                    )
                    self.render_points(
                        "contact_points1",
                        self.contact_points1.numpy(),
                        radius=self.contact_points_radius * self.scaling,
                        colors=self.contact_points1_colors,
                    )

    return SimRenderer


SimRendererUsd = CreateSimRenderer(wp.render.UsdRenderer)
SimRendererOpenGL = CreateSimRenderer(wp.render.OpenGLRenderer)
SimRenderer = SimRendererUsd
