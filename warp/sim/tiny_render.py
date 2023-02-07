# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import sys

import warp as wp
import warp.sim

from collections import defaultdict

import numpy as np


def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), upaxis="y"):
    # compute positional offsets per environment
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs**(1.0/num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i*env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length*side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(upaxis, str):
        upaxis = "xyz".index(upaxis.lower())
    correction[upaxis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets

@wp.kernel
def update_vbo(
    shape_ids: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    instance_envs: wp.array(dtype=int),
    env_offsets: wp.array(dtype=wp.vec3),
    scaling: float,
    bodies_per_env: int,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4),
    vbo_orientations: wp.array(dtype=wp.quat)):

    tid = wp.tid()
    shape = shape_ids[tid]
    body = shape_body[shape]
    env = instance_envs[tid]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = body_q[body+env*bodies_per_env] * X_ws
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    p *= scaling
    p += env_offsets[env]
    vbo_positions[tid] = wp.vec4(p[0], p[1], p[2], 0.0)
    vbo_orientations[tid] = q

class TinyRenderer:
    
    def __init__(
        self,
        model: warp.sim.Model,
        title="Warp sim",
        scaling=1.0,
        fps=60,
        upaxis="y",
        env_offset=(5.0, 0.0, 5.0),
        suppress_keyboard_help=False,
        start_paused=False):

        try:
            import pytinyopengl3 as p
        except ImportError:
            print("pytinyopengl3 not found, it can be installed via `pip install pytinydiffsim`")
            raise

        self.paused = False
        self.skip_rendering = False
        self._skip_frame_counter = 0

        self.app = p.TinyOpenGL3App(title)
        self.app.renderer.init()
        def keypress(key, pressed):
            if not pressed:
                return
            if key == 27:  # ESC
                self.app.window.set_request_exit()
                sys.exit(0)
            if key == 32:  # space
                self.paused = not self.paused
            if key == ord('s'):
                self.skip_rendering = not self.skip_rendering

        self.app.window.set_keyboard_callback(keypress)
        self.cam = p.TinyCamera()
        self.cam.set_camera_distance(25.)
        self.cam.set_camera_pitch(-20)
        self.cam.set_camera_yaw(225)
        self.cam.set_camera_target_position(0.0, 0.0, 0.0)
        self.cam_axis = "xyz".index(upaxis.lower())
        self.cam.set_camera_up_axis(self.cam_axis)
        self.app.renderer.set_camera(self.cam)

        self.model = model
        self.num_envs = model.num_envs
        self.shape_body = model.shape_body.numpy()
        self.shape_transform = model.shape_transform.numpy()

        self.scaling = scaling

        # mapping from instance to shape ID
        self.instance_shape = []
        # mapping from hash of geometry to shape ID
        self.geo_shape = {}

        # first assemble all instances by shape, so we can send instancing commands in bulk for each shape
        # tinyrenderer doesn't correctly display instances if the coresponding shapes are out of order
        instance_shape = defaultdict(list)
        instance_pos = defaultdict(list)
        instance_orn = defaultdict(list)
        instance_scale = defaultdict(list)
        instance_color = defaultdict(list)

        # render meshes double sided
        double_sided_meshes = False

        # create rigid shape children
        if (self.model.shape_count):
            shape_body = model.shape_body.numpy()
            if model.body_count:
                body_q = model.body_q.numpy()
            else:
                body_q = np.zeros((0, 7), dtype=np.float32)
            shape_geo_src = model.shape_geo_src
            shape_geo_type = model.geo_params.type.numpy()
            shape_geo_scale = model.geo_params.scale.numpy()
            shape_transform = model.shape_transform.numpy()

            # matplotlib "tab10" colors
            colors = [
                [ 31, 119, 180],
                [255, 127,  14],
                [ 44, 160,  44],
                [214,  39,  40],
                [148, 103, 189],
                [140,  86,  75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189,  34],
                [ 23, 190, 207]]
            num_colors = len(colors)

            # loop over shapes excluding the ground plane
            num_shapes = (model.shape_count-1) // self.num_envs
            for s in range(num_shapes):
                geo_type = shape_geo_type[s]
                geo_scale = shape_geo_scale[s] * self.scaling
                geo_src = shape_geo_src[s]
                color = colors[len(self.geo_shape)%num_colors]

                # shape transform in body frame
                body = shape_body[s]
                if body > -1:
                    X_ws = wp.transform_expand(wp.mul(body_q[body], shape_transform[s]))
                else:
                    X_ws = wp.transform_expand(shape_transform[s])
                scale = np.ones(3)
                # check whether we can instance an already created shape with the same geometry
                geo_hash = hash((int(geo_type), geo_src, geo_scale[0], geo_scale[1], geo_scale[2]))
                
                if (geo_type == warp.sim.GEO_PLANE):
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        texture = self.create_check_texture(256, 256, color1=color)
                        faces = [0, 1, 2, 2, 3, 0]
                        normal = (0.0, 1.0, 0.0)
                        width = (geo_scale[0] if geo_scale[0] > 0.0 else 100.0)
                        length = (geo_scale[1] if geo_scale[1] > 0.0 else 100.0)
                        aspect = width / length
                        u = width / scaling * aspect
                        v = length / scaling
                        gfx_vertices = [
                            -width, 0.0, -length, 0.0, *normal, 0.0, 0.0,
                            -width, 0.0,  length, 0.0, *normal, 0.0, v,
                            width, 0.0,  length, 0.0, *normal, u, v,
                            width, 0.0, -length, 0.0, *normal, u, 0.0,
                        ]
                        shape = self.app.renderer.register_shape(gfx_vertices, faces, texture, double_sided_meshes)

                elif (geo_type == warp.sim.GEO_SPHERE):
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        texture = self.create_check_texture(color1=color)
                        shape = self.app.register_graphics_unit_sphere_shape(p.EnumSphereLevelOfDetail.SPHERE_LOD_HIGH, texture)
                    scale *= float(geo_scale[0]) * 2.0  # diameter

                elif (geo_type == warp.sim.GEO_CAPSULE or geo_type == warp.sim.GEO_CYLINDER):
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        radius = float(geo_scale[0])
                        half_height = float(geo_scale[1])
                        up_axis = 1
                        texture = self.create_check_texture(color1=color)
                        shape = self.app.register_graphics_capsule_shape(radius, half_height, up_axis, texture)

                elif (geo_type == warp.sim.GEO_BOX):
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        texture = self.create_check_texture(color1=color)
                        shape = self.app.register_cube_shape(geo_scale[0], geo_scale[1], geo_scale[2], texture, 4)

                elif (geo_type == warp.sim.GEO_MESH):
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        texture = self.create_check_texture(1, 1, color1=color, color2=color)
                        faces = np.array(geo_src.indices).reshape((-1, 3))
                        vertices = np.array(geo_src.vertices)
                        # convert vertices to (x,y,z,w, nx,ny,nz, u,v) format
                        gfx_vertices = np.zeros((len(faces)*3, 9))
                        gfx_indices = np.arange(len(faces)*3).reshape((-1, 3))
                        # compute vertex normals
                        for i, f in enumerate(faces):
                            v0 = vertices[f[0]] * geo_scale
                            v1 = vertices[f[1]] * geo_scale
                            v2 = vertices[f[2]] * geo_scale
                            gfx_vertices[i*3+0, :3] = v0
                            gfx_vertices[i*3+1, :3] = v1
                            gfx_vertices[i*3+2, :3] = v2
                            n = np.cross(v1-v0, v2-v0)
                            gfx_vertices[i*3:i*3+3, 4:7] = n / np.linalg.norm(n)
                            
                        shape = self.app.renderer.register_shape(
                            gfx_vertices.flatten(),
                            gfx_indices.flatten(),
                            texture,
                            double_sided_meshes)

                elif (geo_type == warp.sim.GEO_SDF):
                    continue
                else:
                    print("Unknown geometry type: ", geo_type)
                    continue

                if geo_hash not in self.geo_shape:
                    self.geo_shape[geo_hash] = shape

                instance_shape[shape].append(s)
                instance_pos[shape].append(p.TinyVector3f(*X_ws.p))
                instance_orn[shape].append(p.TinyQuaternionf(*X_ws.q))
                instance_scale[shape].append(p.TinyVector3f(*scale))
                instance_color[shape].append(p.TinyVector3f(1.,1.,1.))

        # create instances for each shape
        for i, shape in enumerate(instance_shape.keys()):
            opacity = 1
            rebuild = (i == len(instance_shape)-1)
            self.instance_shape.extend(np.repeat(instance_shape[shape], self.num_envs))
            self.app.renderer.register_graphics_instances(
                shape,
                instance_pos[shape] * self.num_envs,
                instance_orn[shape] * self.num_envs,
                instance_color[shape] * self.num_envs,
                instance_scale[shape] * self.num_envs,
                opacity, rebuild
            )

        if model.ground:
            color1 = (200, 200, 200)
            color2 = (150, 150, 150)
            texture = self.create_check_texture(256, 256, color1=color1, color2=color2)
            faces = [0, 1, 2, 2, 3, 0]
            normal = (0.0, 1.0, 0.0)
            geo_scale = shape_geo_scale[-1]
            width = 100.0 * scaling
            length = 100.0 * scaling
            u = 100.0
            v = 100.0
            gfx_vertices = [
                -width, 0.0, -length, 0.0, *normal, 0.0, 0.0,
                -width, 0.0,  length, 0.0, *normal, 0.0, v,
                 width, 0.0,  length, 0.0, *normal, u, v,
                 width, 0.0, -length, 0.0, *normal, u, 0.0,
            ]
            shape = self.app.renderer.register_shape(gfx_vertices, faces, texture, double_sided_meshes)
            X_ws = wp.transform_expand(shape_transform[-1])
            pos = p.TinyVector3f(*X_ws.p)
            orn = p.TinyQuaternionf(*X_ws.q)
            color = p.TinyVector3f(1.,1.,1.)
            scale = p.TinyVector3f(1.,1.,1.)
            opacity = 1
            rebuild = True
            self.app.renderer.register_graphics_instance(shape, pos, orn, color, scale, opacity, rebuild)

        self.app.renderer.write_transforms()
        
        self.num_instances = len(self.instance_shape)
        self.bodies_per_env = len(self.model.body_q) // self.num_envs
        self.instances_per_env = self.num_instances // self.num_envs
        
        # mapping from shape instance to environment ID
        self.instance_envs = wp.array(
            np.tile(np.arange(self.num_envs, dtype=np.int32), self.instances_per_env), dtype=wp.int32,
            device="cuda", owner=False, ndim=1)
        
        env_offsets = compute_env_offsets(self.num_envs, env_offset, self.cam_axis)
        self.env_offsets = wp.array(env_offsets * scaling, dtype=wp.vec3, device="cuda")
        self.instance_shape = wp.array(self.instance_shape, dtype=wp.int32, device="cuda")
        # make sure the static arrays are on the GPU
        if self.model.shape_transform.device.is_cuda:
            self.shape_transform = self.model.shape_transform
            self.shape_body = self.model.shape_body
        else:
            self.shape_transform = self.model.shape_transform.to("cuda")
            self.shape_body = self.model.shape_body.to("cuda")

        if not suppress_keyboard_help:
            print("Control commands for the TinyRenderer window:")
            print("  [Space]                                   pause simulation")
            print("  [S]                                       skip rendering")
            print("  [Alt] + mouse drag (left/middle button)   rotate/pan camera")
            print("  [ESC]                                     exit")

        if start_paused:
            self.begin_frame(0.0)
            self.render(model.state())
            self.end_frame()
            self.paused = True

    def render(self, state: warp.sim.State):
        if self.skip_rendering:
            return

        if (self.model.particle_count):
            pass

            # particle_q = state.particle_q.numpy()

            # # render particles
            # self.render_points("particles", particle_q, radius=self.model.soft_contact_distance)

            # # render tris
            # if (self.model.tri_count):
            #     self.render_mesh("surface", particle_q, self.model.tri_indices.numpy().flatten())

            # # render springs
            # if (self.model.spring_count):
            #     self.render_line_list("springs", particle_q, self.model.spring_indices.numpy().flatten(), [], 0.1)

        # render muscles
        if (self.model.muscle_count):
            pass
            
            # body_q = state.body_q.numpy()

            # muscle_start = self.model.muscle_start.numpy()
            # muscle_links = self.model.muscle_bodies.numpy()
            # muscle_points = self.model.muscle_points.numpy()
            # muscle_activation = self.model.muscle_activation.numpy()

            # for m in range(self.model.muscle_count):

            #     start = int(muscle_start[m])
            #     end = int(muscle_start[m + 1])

            #     points = []

            #     for w in range(start, end):
                    
            #         link = muscle_links[w]
            #         point = muscle_points[w]

            #         X_sc = wp.transform_expand(body_q[link][0])

            #         points.append(Gf.Vec3f(wp.transform_point(X_sc, point).tolist()))
                
            #     self.render_line_strip(name=f"muscle_{m}", vertices=points, radius=0.0075, color=(muscle_activation[m], 0.2, 0.5))
        
        

        # update bodies
        if (self.model.body_count):
            
            wp.synchronize()
            if state.body_q.device.is_cuda:
                self.body_q = state.body_q
            else:
                self.body_q = state.body_q.to("cuda")

            vbo = self.app.cuda_map_vbo()
            vbo_positions = wp.array(
                ptr=vbo.positions, dtype=wp.vec4, shape=(self.num_instances,),
                length=self.num_instances, capacity=self.num_instances,
                device="cuda", owner=False, ndim=1)
            vbo_orientations = wp.array(
                ptr=vbo.orientations, dtype=wp.quat, shape=(self.num_instances,),
                length=self.num_instances, capacity=self.num_instances,
                device="cuda", owner=False, ndim=1)
            wp.launch(
                update_vbo,
                dim=self.num_instances,
                inputs=[
                    self.instance_shape,
                    self.shape_body,
                    self.shape_transform,
                    self.body_q,
                    self.instance_envs,
                    self.env_offsets,
                    self.scaling,
                    self.bodies_per_env,
                ],
                outputs=[
                    vbo_positions,
                    vbo_orientations,
                ],
                device="cuda")
            self.app.cuda_unmap_vbo()

    def begin_frame(self, time: float):
        self.time = time
        while self.paused and not self.app.window.requested_exit():
            self.update()
        if self.app.window.requested_exit():
            sys.exit(0)

    def end_frame(self):
        self.update()

    def update(self):
        self._skip_frame_counter += 1
        if self._skip_frame_counter > 100:
            self._skip_frame_counter = 0
        if self.skip_rendering:
            if self._skip_frame_counter == 0:
                # ensure we receive key events
                self.app.swap_buffer()
            return
        self.app.renderer.update_camera(self.cam_axis)
        self.app.renderer.render_scene()
        self.app.swap_buffer()

    def save(self):
        while not self.app.window.requested_exit():
            self.update()
        if self.app.window.requested_exit():
            sys.exit(0)

    def create_check_texture(self, width=256, height=256, color1=(0, 128, 256), color2=None):
        pixels = np.zeros((width, height, 3), dtype=np.uint8)
        half_w = width // 2
        half_h = height // 2
        pixels[0:half_w, 0:half_h] = color1
        pixels[half_w:width, half_h:height] = color1
        if color2 is None:
            color2 = np.clip(np.array(color1) + 50, 0, 255)
        pixels[half_w:width, 0:half_h] = color2
        pixels[0:half_w, half_h:height] = color2
        return self.app.renderer.register_texture(pixels.flatten().tolist(), width, height, False)
