# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os

import warp as wp
import warp.sim
from .utils import mul_elemwise, tab10_color_map

from collections import defaultdict
from typing import Union

import numpy as np


@wp.kernel
def update_vbo_transforms(
    instance_id: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    scaling: float,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4),
    vbo_orientations: wp.array(dtype=wp.quat)):

    shape = wp.tid()
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        if body_q:
            X_ws = body_q[body] * X_ws
        else:
            return
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    p *= scaling
    i = instance_id[shape]
    vbo_positions[i] = wp.vec4(p[0], p[1], p[2], 0.0)
    vbo_orientations[i] = q


@wp.kernel
def update_points_positions(
    instance_id: wp.array(dtype=int),
    position: wp.array(dtype=wp.vec3),
    scaling: float,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4)):

    tid = wp.tid()
    p = position[tid] * scaling
    vbo_positions[instance_id[tid]] = wp.vec4(p[0], p[1], p[2], 0.0)


@wp.kernel
def update_line_transforms(
    instance_id: wp.array(dtype=int),
    lines: wp.array(dtype=wp.vec3, ndim=2),
    scaling: float,
    # outputs
    vbo_positions: wp.array(dtype=wp.vec4),
    vbo_orientations: wp.array(dtype=wp.quat),
    vbo_scalings: wp.array(dtype=wp.vec4)):

    tid = wp.tid()
    p0 = lines[tid, 0]
    p1 = lines[tid, 1]
    p = (p0 + p1) * (0.5 * scaling)
    d = p1 - p0
    s = wp.length(d)
    axis = wp.normalize(d)
    y_up = wp.vec3(0.0, 1.0, 0.0)
    angle = wp.acos(wp.dot(axis, y_up))
    axis = wp.normalize(wp.cross(axis, y_up))
    q = wp.quat_from_axis_angle(axis, -angle)
    i = instance_id[tid]
    vbo_positions[i] = wp.vec4(p[0], p[1], p[2], 0.0)
    vbo_orientations[i] = q
    vbo_scalings[i] = wp.vec4(1.0, s, 1.0, 1.0)


# convert mesh into TinyRenderer-compatible vertex buffer
@wp.kernel
def compute_gfx_vertices(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    geo_scale: wp.vec3,
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2)):

    tid = wp.tid()
    v0 = mul_elemwise(vertices[indices[tid, 0]], geo_scale)
    v1 = mul_elemwise(vertices[indices[tid, 1]], geo_scale)
    v2 = mul_elemwise(vertices[indices[tid, 2]], geo_scale)
    i = tid * 3; j = i + 1; k = i + 2
    gfx_vertices[i,0] = v0[0]; gfx_vertices[i,1] = v0[1]; gfx_vertices[i,2] = v0[2]
    gfx_vertices[j,0] = v1[0]; gfx_vertices[j,1] = v1[1]; gfx_vertices[j,2] = v1[2]
    gfx_vertices[k,0] = v2[0]; gfx_vertices[k,1] = v2[1]; gfx_vertices[k,2] = v2[2]
    n = wp.normalize(wp.cross(v1-v0, v2-v0))
    gfx_vertices[i,4] = n[0]; gfx_vertices[i,5] = n[1]; gfx_vertices[i,6] = n[2]
    gfx_vertices[j,4] = n[0]; gfx_vertices[j,5] = n[1]; gfx_vertices[j,6] = n[2]
    gfx_vertices[k,4] = n[0]; gfx_vertices[k,5] = n[1]; gfx_vertices[k,6] = n[2]


@wp.kernel
def update_gfx_vertices(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    geo_scale: wp.vec3,
    offset: int,
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2)):

    tid = wp.tid()
    v0 = mul_elemwise(vertices[indices[tid, 0]], geo_scale)
    v1 = mul_elemwise(vertices[indices[tid, 1]], geo_scale)
    v2 = mul_elemwise(vertices[indices[tid, 2]], geo_scale)
    i = tid * 3 + offset; j = i + 1; k = i + 2
    gfx_vertices[i,0] = v0[0]; gfx_vertices[i,1] = v0[1]; gfx_vertices[i,2] = v0[2]
    gfx_vertices[j,0] = v1[0]; gfx_vertices[j,1] = v1[1]; gfx_vertices[j,2] = v1[2]
    gfx_vertices[k,0] = v2[0]; gfx_vertices[k,1] = v2[1]; gfx_vertices[k,2] = v2[2]


class TinyRenderer:

    # number of segments to use for rendering cones and cylinders
    default_num_segments = 64
    # number of horizontal and vertical pixels to use for checkerboard texture
    default_texture_size = 256
    
    # render meshes double sided
    double_sided_meshes = False

    default_points_color = (48/255, 171/255, 242/255)

    def __init__(
        self,
        title="Warp sim",
        scaling=1.0,
        fps=60,
        upaxis="y",
        suppress_keyboard_help=False,
        move_camera_target_to_center=True,
        screen_width=1024,
        screen_height=768,
        headless=False):

        try:
            import pytinyopengl3 as p
            self.p = p
        except ImportError:
            print("pytinyopengl3 not found, it can be installed via `pip install pytinydiffsim`")
            raise

        self.paused = False
        self.skip_rendering = False
        self._skip_frame_counter = 0
        self.scaling = scaling

        if title.endswith(".usd"):
            title = os.path.basename(title)[:-4]
        if headless:
            window_type = 2  # use EGL
        else:
            window_type = 0
        self.app = p.TinyOpenGL3App(title, width=screen_width, height=screen_height, windowType=window_type)
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
        cam_pos = np.zeros(3)
        # if move_camera_target_to_center and len(model.body_q):
        #     cam_pos = model.body_q.numpy()[:, :3].mean(axis=0) * scaling
        self.cam.set_camera_target_position(*cam_pos)
        self.cam.set_camera_distance(25.)
        self.cam.set_camera_pitch(-20)
        self.cam.set_camera_yaw(225)
        self.cam_axis = "xyz".index(upaxis.lower())
        self.cam.set_camera_up_axis(self.cam_axis)
        self.app.renderer.set_camera(self.cam)

        self._instance_count = 0
        
        # first assemble all instances by shape, so we can send instancing commands in bulk for each shape
        # tinyrenderer doesn't correctly display instances if the coresponding shapes are out of order
        self._shape_instance_body = defaultdict(list)
        self._shape_instance_pos = defaultdict(list)
        self._shape_instance_rot = defaultdict(list)
        self._shape_instance_scale = defaultdict(list)
        self._shape_instance_color = defaultdict(list)
        self._shape_instance_created = defaultdict(list)
        self._shape_instance_name = defaultdict(list)  # mapping from instance name to shape ID and instance index within the same shape

        # keep track of order of when shapes were added
        self._shape_order = []
        # mapping from shape ID to its instances (indices of when instances were added)
        self._shape_instances = defaultdict(list)

        self._shape_count = 0
        self._shape_transform = []
        self._shape_scale = []
        self._shape_body = []
        self._shape_transform_wp = None
        self._shape_scale_wp = None
        self._shape_body_wp = None
        self._shape_transform_index = {}  # mapping from instance name to index of transform
        self._shape_instance_name_mapping = {}  # mapping from instance name to shape ID and instance index within the same shape
        self._shape_name = {}  # mapping from shape name to shape ID
        self._shape_scale_ref = {}  # reference scale of each shape
        # instance IDs in the renderer
        self._shape_instance_ids = []
        self._shape_instance_ids_wp = None
        # mapping from hash of geometry to shape ID
        self._shape_geo_hash = {}

        self._mesh_name = {}  # mapping from mesh name to shape ID

        self._body_name = {}  # mapping from body name to its body ID
        self._body_shapes = defaultdict(list)  # mapping from body index to its shape IDs

        self._has_new_instances = False
        # updates to instances applied during `self.end_frame()` (instance ID, transform, scale, color)
        self._shape_instance_updates = []

        # instance IDs of point spheres
        self._point_instance_ids = defaultdict(list)
        self._point_instance_ids_wp = {}  # warp arrays
        # shape ID of particle spheres
        self._point_shape = {}

        # mapping from name to instances of capsules
        self._line_instance_ids = defaultdict(list)
        self._line_instance_ids_wp = {}  # warp arrays
        self._line_shape = {}  # mapping from name to ID of capsule shape
    
        if not headless and not suppress_keyboard_help:
            print("Control commands for the TinyRenderer window:")
            print("  [Space]                                   pause simulation")
            print("  [S]                                       skip rendering")
            print("  [Alt] + mouse drag (left/middle button)   rotate/pan camera")
            print("  [ESC]                                     exit")

    def clear(self):
        self._shape_instance_body.clear()
        self._shape_instance_pos.clear()
        self._shape_instance_rot.clear()
        self._shape_instance_scale.clear()
        self._shape_instance_color.clear()
        self._shape_instance_created.clear()
        self._shape_instance_name.clear()
        self._shape_order.clear()
        self._shape_instances.clear()
        self._shape_count = 0
        self._shape_transform.clear()
        self._shape_scale.clear()
        self._shape_body.clear()
        self._shape_transform_wp = None
        self._shape_scale_wp = None
        self._shape_body_wp = None
        self._shape_transform_index.clear()
        self._shape_instance_name_mapping.clear()
        self._shape_name.clear()
        self._shape_scale_ref.clear()
        self._shape_geo_hash.clear()
        self._mesh_name.clear()
        self._body_name.clear()
        self._body_shapes.clear()
        self._has_new_instances = False
        self._shape_instance_updates.clear()
        self._point_instance_ids.clear()
        self._point_instance_ids_wp.clear()
        self._point_shape.clear()
        self._line_instance_ids.clear()
        self._line_instance_ids_wp.clear()
        self._line_shape.clear()
        self._instance_count = 0
        self.app.renderer.remove_all_instances()

    def register_body(self, name):
        # register body name and return its ID
        if name not in self._body_name:
            self._body_name[name] = len(self._body_name)
        return self._body_name[name]

    def _resolve_body_id(self, body):
        if body is None:
            return -1
        if isinstance(body, int):
            return body
        return self._body_name[body]
    
    def add_shape_instance(self, name: str, shape: int, body: int, pos: tuple, rot: tuple, scale: tuple=(1.,1.,1.), color: tuple=(1.,1.,1.)):
        # plan to add a new shape instance, to be created during `complete_setup()`
        self._shape_instance_name[shape].append(name)
        self._shape_instance_body[shape].append(self._resolve_body_id(body))
        self._shape_instance_pos[shape].append(pos)
        self._shape_instance_rot[shape].append(rot)
        shape_scale = self._shape_scale_ref[shape]
        self._shape_instance_scale[shape].append((scale[0]*shape_scale[0], scale[1]*shape_scale[1], scale[2]*shape_scale[2]))
        self._shape_instance_color[shape].append((1.,1.,1.))
        self._shape_instance_created[shape].append(False)
        self._has_new_instances = True
    
    def _get_new_color(self):
        return tab10_color_map(self._shape_geo_hash)

    def _add_shape(self, shape: int, name: str, scale: tuple=(1.,1.,1.)):
        # register shape name and reference scale to use by the shape instances
        if name not in self._shape_name:
            self._shape_name[name] = shape
            self._shape_order.append((name, shape))
        self._shape_scale_ref[shape] = scale

    def _rebuild_instances(self):
        # reorder shape instance IDs to match the VBO order
        idx = 0
        self._instance_count = self.app.renderer.get_total_num_instances()
        self._shape_instance_ids = np.zeros(self._instance_count, dtype=np.int32)
        for name, shape in self._shape_order:
            idxs = np.arange(idx, idx+len(self._shape_instances[shape]), dtype=np.int32)
            idx += len(self._shape_instances[shape])
            # self._shape_instance_ids.extend(idxs)
            self._shape_instance_ids[idxs] = idxs
            if name in self._line_instance_ids:
                self._line_instance_ids[name] = idxs.tolist()
                self._line_instance_ids_wp[name] = wp.array(idxs, dtype=wp.int32, device="cuda")
            elif name in self._point_instance_ids:
                self._point_instance_ids[name] = idxs.tolist()
                self._point_instance_ids_wp[name] = wp.array(idxs, dtype=wp.int32, device="cuda")
        self._shape_instance_ids_wp = wp.array(self._shape_instance_ids, dtype=wp.int32, device="cuda")
        self._shape_instance_ids = self._shape_instance_ids.tolist()

    def _add_instances(self, shape, pos, rot, color, scale, opacity=1., rebuild=True):
        # add shape instances to TinyRenderer and ensure the transforms of the previous instances are preserved
        vbo = self.app.cuda_map_vbo()
        num_instances = self.app.renderer.get_total_num_instances()
        orig_positions = wp.clone(wp.array(
            ptr=vbo.positions, dtype=wp.vec4, shape=(num_instances,),
            device="cuda", owner=False, ndim=1))
        orig_orientations = wp.clone(wp.array(
            ptr=vbo.orientations, dtype=wp.quat, shape=(num_instances,),
            device="cuda", owner=False, ndim=1))
        orig_scalings = wp.clone(wp.array(
            ptr=vbo.scalings, dtype=wp.vec4, shape=(num_instances,),
            device="cuda", owner=False, ndim=1))
        vcnt = self.app.renderer.get_shape_vertex_count()
        total_vertices = sum(vcnt)
        orig_vertices = wp.clone(wp.array(
            ptr=vbo.vertices, dtype=wp.float32, shape=(total_vertices,9),
            device="cuda", owner=False, ndim=2))
        self.app.cuda_unmap_vbo()
        new_ids = self.app.renderer.register_graphics_instances(
            shape, pos, rot, color, scale, opacity, rebuild)
        self._shape_instances[shape].extend(new_ids)
        self.app.renderer.write_transforms()
        # restore the transforms of the previous instances
        vbo = self.app.cuda_map_vbo()
        vbo_positions = wp.array(
            ptr=vbo.positions, dtype=wp.vec4, shape=(num_instances,),
            device="cuda", owner=False, ndim=1)
        vbo_orientations = wp.array(
            ptr=vbo.orientations, dtype=wp.quat, shape=(num_instances,),
            device="cuda", owner=False, ndim=1)
        vbo_scalings = wp.array(
            ptr=vbo.scalings, dtype=wp.vec4, shape=(num_instances,),
            device="cuda", owner=False, ndim=1)
        vbo_vertices = wp.array(
            ptr=vbo.vertices, dtype=wp.float32, shape=(total_vertices,9),
            device="cuda", owner=False, ndim=2)
        vbo_positions.assign(orig_positions)
        vbo_orientations.assign(orig_orientations)
        vbo_scalings.assign(orig_scalings)
        # vbo_vertices.assign(orig_vertices)
        self.app.cuda_unmap_vbo()

        if rebuild:
            self._rebuild_instances()

        # self._instance_count += len(new_ids)
        self._instance_count = self.app.renderer.get_total_num_instances()
        # print(f"shape {shape}\tnew_ids:", new_ids)
        # if len(orig_positions) > 0:
        #     print("pos:", orig_positions.numpy())
        return new_ids
    
    def _add_mesh(self, faces, vertices, geo_scale, texture, double_sided_meshes=double_sided_meshes):
        # convert vertices to (x,y,z,w, nx,ny,nz, u,v) format
        gfx_vertices = wp.zeros((len(faces)*3, 9), dtype=float)
        gfx_indices = np.arange(len(faces)*3).reshape((-1, 3))
        wp.launch(
            compute_gfx_vertices,
            dim=len(faces),
            inputs=[
                wp.array(faces, dtype=int),
                wp.array(vertices, dtype=wp.vec3),
                wp.vec3(*geo_scale)],
            outputs=[gfx_vertices])
        gfx_vertices = gfx_vertices.numpy()
        return self.app.renderer.register_shape(
            gfx_vertices.flatten(),
            gfx_indices.flatten(),
            texture,
            double_sided_meshes)
    
    def _update_mesh(self, name, vertices, indices, scale: tuple=None):
        # update mesh vertices
        vcnt = self.app.renderer.get_shape_vertex_count()
        voffsets = self.app.renderer.get_shape_vertex_offsets()
        
        idx = self._shape_transform_index[name]
        shape = self._mesh_name[name]
        total_vertices = sum(vcnt)
        offset = voffsets[shape]
        
        if isinstance(vertices, wp.array):
            vertices_dest = vertices
            if not vertices_dest.device.is_cuda:
                vertices_dest = vertices_dest.to("cuda")
        else:
            vertices_dest = wp.array(
                vertices,
                dtype=wp.vec3,
                device="cuda")
        
        faces = wp.array(np.array(indices).reshape(-1, 3), dtype=int, device="cuda")
        vbo = self.app.cuda_map_vbo()
        vbo_vertices = wp.array(
            ptr=vbo.vertices, dtype=wp.float32, shape=(total_vertices,9),
            device="cuda", owner=False, ndim=2)
        if scale is None:
            geo_scale = wp.vec3(*self._shape_scale[shape])
        else:
            geo_scale = wp.vec3(scale[0]*self.scaling, scale[1]*self.scaling, scale[2]*self.scaling)
            
        wp.launch(
            update_gfx_vertices,
            device="cuda",
            dim=len(faces),
            inputs=[faces, vertices_dest, geo_scale, offset],
            outputs=[vbo_vertices])

        self.app.cuda_unmap_vbo()
    
    def _update_instance(self, instance_name: str, pos: tuple, rot: tuple, scale: tuple=None):
        # update transform and scale of a shape instance with the given name
        if instance_name not in self._shape_transform_index:
            return False
        idx = self._shape_transform_index[instance_name]
        self._shape_instance_updates.append((idx, (*pos, *rot), scale))
        return True

    def complete_setup(self):
        # create instances for each shape
        added_instances = False
        shapes = list(self._shape_instance_created.keys())
        for shape in shapes:
            pos = []; rot = []; color = []; scale = []
            data = zip(
                self._shape_instance_created[shape],
                self._shape_instance_name[shape],
                self._shape_instance_pos[shape],
                self._shape_instance_rot[shape],
                self._shape_instance_color[shape],
                self._shape_instance_scale[shape],
                self._shape_instance_body[shape])
            for i, (created, name, p, q, c, s, b) in enumerate(data):
                if created:
                    continue
                pos.append(self.p.TinyVector3f(p[0]*self.scaling, p[1]*self.scaling, p[2]*self.scaling))
                rot.append(self.p.TinyQuaternionf(*q))
                color.append(self.p.TinyVector3f(*c))
                scale.append(self.p.TinyVector3f(*s))
                self._shape_transform_index[name] = len(self._shape_transform)
                self._shape_transform.append((*p, *q))
                self._shape_scale.append(s)
                self._shape_body.append(b)
            if len(pos) > 0:
                new_ids = self._add_instances(shape, pos, rot, color, scale)
                for i, name in zip(new_ids, self._shape_instance_name[shape]):
                    if name is not None:
                        self._shape_instance_name_mapping[name] = i
                self._shape_instance_ids.extend(new_ids)
                added_instances = True
                
            # self._shape_instance_created[shape] = [True] * len(self._shape_instance_created[shape])
            del self._shape_instance_created[shape]
            del self._shape_instance_name[shape]
            del self._shape_instance_pos[shape]
            del self._shape_instance_rot[shape]
            del self._shape_instance_color[shape]
            del self._shape_instance_scale[shape]
            del self._shape_instance_body[shape]

        if added_instances:
            self._shape_instance_ids_wp = wp.array(self._shape_instance_ids, dtype=int, device="cuda")

        self._shape_count = len(self._shape_transform)
        self._shape_transform_wp = wp.array(self._shape_transform, dtype=wp.transform, device="cuda")
        self._shape_scale_wp = wp.array(self._shape_scale, dtype=wp.vec3, device="cuda")
        self._shape_body_wp = wp.array(self._shape_body, dtype=int, device="cuda")
        self._has_new_instances = False
    
    @property
    def projection_matrix(self):
        return np.array(self.cam.get_camera_projection_matrix()).reshape((4,4))
    
    @property
    def view_matrix(self):
        return np.array(self.cam.get_camera_view_matrix()).reshape((4,4))

    @property
    def screen_width(self):
        return self.app.renderer.get_screen_width()

    @property
    def screen_height(self):
        return self.app.renderer.get_screen_height()

    def move_camera(self, target_position: tuple, distance: float, yaw: float, pitch: float):
        self.cam.set_camera_target_position(*target_position)
        self.cam.set_camera_distance(distance)
        self.cam.set_camera_yaw(yaw)
        self.cam.set_camera_pitch(pitch)

    def get_pixel_buffer(self):
        self.app.renderer.render_scene()
        pixels = self.p.ReadPixelBuffer(self.app)
        img = np.reshape(pixels.rgba, (self.screen_height, self.screen_width, 4))
        img = img / 255.
        img = np.flipud(img)
        return img

    def _resolve_body(self, body):
        # resolve body ID from name, or return the ID directly
        if body is None:
            return -1
        if isinstance(body, str):
            return self._body_name[body]
        return body

    def save_frame_to_png(self, filename: str, render_width: int = None, render_height: int = None):
        render_width = render_width or self.screen_width
        render_height = render_height or self.screen_height
        render_to_texture = (render_width != self.screen_width or render_height != self.screen_height)
        self.app.dump_next_frame_to_png(filename, render_to_texture, render_width, render_height)

    def save_frames_to_video(self, filename: str):
        # save frames to a MP4 video file; ffmpeg must be installed and available in the PATH
        self.app.dump_frames_to_video(filename)

    def render_plane(self, name: str, pos: tuple, rot: tuple, width: float, length: float, color: tuple=(1.,1.,1.), texture=None, parent_body: str=None, is_template: bool=False, u_scaling=1.0, v_scaling=1.0):
        """Add a plane for visualization
        
        Args:
            name: The name of the plane
            pos: The position of the plane
            rot: The rotation of the plane
            width: The width of the plane
            length: The length of the plane
            color: The color of the plane
            texture: The texture of the plane (optional)
        """
        geo_hash = hash((int(warp.sim.GEO_PLANE), width, length))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self._update_instance(name, pos, rot):
                return shape
        else:
            if texture is None:
                texture = self.create_check_texture(color1=color)
            faces = [0, 1, 2, 2, 3, 0]
            normal = (0.0, 1.0, 0.0)
            width = (width if width > 0.0 else 100.0) * self.scaling
            length = (length if length > 0.0 else 100.0) * self.scaling
            aspect = width / length
            u = width * aspect * u_scaling / self.scaling
            v = length * v_scaling / self.scaling
            gfx_vertices = [
                -width, 0.0, -length, 0.0, *normal, 0.0, 0.0,
                -width, 0.0,  length, 0.0, *normal, 0.0, v,
                width, 0.0,  length, 0.0, *normal, u, v,
                width, 0.0, -length, 0.0, *normal, u, 0.0,
            ]
            double_sided_meshes = False
            shape = self.app.renderer.register_shape(gfx_vertices, faces, texture, double_sided_meshes)
            self._add_shape(shape, name)
            self._shape_geo_hash[geo_hash] = shape
        if not is_template:
            body = self._resolve_body(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, (1.0, 1.0, 1.0), (1., 1., 1.))
        return shape

    def render_ground(self, size: float=100.0):
        """Add a ground plane for visualization
        
        Args:
            size: The size of the ground plane
        """
        color1 = (200/255, 200/255, 200/255)
        color2 = (150/255, 150/255, 150/255)
        texture = self.create_check_texture(color1=color1, color2=color2)
        return self.render_plane("ground", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), size, size, texture=texture, u_scaling=1.0, v_scaling=1.0)
    
    def render_sphere(self, name: str, pos: tuple, rot: tuple, radius: float, parent_body: str=None, is_template: bool=False):
        """Add a sphere for visualization
        
        Args:
            pos: The position of the sphere
            radius: The radius of the sphere
            name: A name for the USD prim on the stage
        """
        geo_hash = hash((int(warp.sim.GEO_SPHERE), radius))
        scale = float(radius) * 2. * self.scaling
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self._update_instance(name, pos, rot, (scale, scale, scale)):
                return shape
        else:
            texture = self.create_check_texture(color1=self._get_new_color())
            shape = self.app.register_graphics_unit_sphere_shape(self.p.EnumSphereLevelOfDetail.SPHERE_LOD_HIGH, texture)
            self._add_shape(shape, name, (scale, scale, scale))
            self._shape_geo_hash[geo_hash] = shape
        if not is_template:
            body = self._resolve_body(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_capsule(self, name: str, pos: tuple, rot: tuple, radius: float, half_height: float, parent_body: str=None, is_template: bool=False):
        """Add a capsule for visualization
        
        Args:
            pos: The position of the capsule
            radius: The radius of the capsule
            half_height: The half height of the capsule
            name: A name for the USD prim on the stage
        """
        geo_hash = hash((int(warp.sim.GEO_CAPSULE), radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self._update_instance(name, pos, rot):
                return shape
        else:
            texture = self.create_check_texture(color1=self._get_new_color())
            up_axis = 1
            shape = self.app.register_graphics_capsule_shape(radius * self.scaling, half_height * self.scaling, up_axis, texture)
            self._add_shape(shape, name)
            self._shape_geo_hash[geo_hash] = shape
        if not is_template:
            body = self._resolve_body(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, (1., 1., 1.))
        return shape
    
    def render_cylinder(self, name: str, pos: tuple, rot: tuple, radius: float, half_height: float, parent_body: str=None, is_template: bool=False):
        """Add a cylinder for visualization
        
        Args:
            pos: The position of the cylinder
            radius: The radius of the cylinder
            half_height: The half height of the cylinder
            name: A name for the USD prim on the stage
        """
        geo_hash = hash((int(warp.sim.GEO_CYLINDER), radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self._update_instance(name, pos, rot):
                return shape
        else:
            texture = self.create_check_texture(color1=self._get_new_color())
            
            k = self.default_num_segments
            ts = np.linspace(0, 2*np.pi, k)
            base_x = np.sin(ts) * radius
            base_z = np.cos(ts) * radius
            bot_vs = np.stack([base_x, [-half_height]*k, base_z], axis=1)
            top_vs = np.stack([base_x, [half_height]*k, base_z], axis=1)

            vertices = np.vstack((bot_vs, top_vs, [[0.0, -half_height, 0.0]], [[0.0, half_height, 0.0]]))
            sides1 = np.array([[i, (i+1)%k, k+i] for i in range(k)])
            sides2 = np.array([[k+i, (i+1)%k, k+(i+1)%k] for i in range(k)])
            bottom = np.array([[(i+1)%k, i, 2*k] for i in range(k)])
            top = np.array([[2*k+1, k+i, k+(i+1)%k] for i in range(k)])
            faces = np.vstack((sides1, sides2, bottom, top))

            shape = self._add_mesh(faces, vertices, np.ones(3), texture)

            self._add_shape(shape, name)
            self._shape_geo_hash[geo_hash] = shape
        if not is_template:
            body = self._resolve_body(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, (1., 1., 1.))
        return shape
    
    def render_cone(self, name: str, pos: tuple, rot: tuple, radius: float, half_height: float, parent_body: str=None, is_template: bool=False):
        """Add a cone for visualization
        
        Args:
            pos: The position of the cone
            radius: The radius of the cone
            half_height: The half height of the cone
            name: A name for the USD prim on the stage
        """
        geo_hash = hash((int(warp.sim.GEO_CONE), radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self._update_instance(name, pos, rot):
                return shape
        else:
            texture = self.create_check_texture(color1=self._get_new_color())
            radius *= self.scaling
            half_height *= self.scaling
            
            k = self.default_num_segments
            ts = np.linspace(0, 2*np.pi, k)
            base_x = np.sin(ts) * radius
            base_z = np.cos(ts) * radius
            base_vs = np.stack([base_x, [-half_height]*k, base_z], axis=1)

            vertices = np.vstack((base_vs, [[0.0, half_height, 0.0]], [[0.0, -half_height, 0.0]]))
            sides = np.array([[i, (i+1)%k, k] for i in range(k)])
            base = np.array([[k+1, (i+1)%k, i] for i in range(k)])
            faces = np.vstack((sides, base))

            shape = self._add_mesh(faces, vertices, np.ones(3), texture)

            self._add_shape(shape, name)
            self._shape_geo_hash[geo_hash] = shape
        if not is_template:
            body = self._resolve_body(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, (1., 1., 1.))
        return shape
    
    def render_box(self, name: str, pos: tuple, rot: tuple, extents: tuple, parent_body: str=None, is_template: bool=False):
        """Add a box for visualization
        
        Args:
            pos: The position of the sphere
            extents: The radius of the sphere
            name: A name for the USD prim on the stage
        """
        geo_hash = hash((int(warp.sim.GEO_BOX), float(extents[0]), float(extents[1]), float(extents[2])))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self._update_instance(name, pos, rot):
                return shape
        else:
            texture = self.create_check_texture(color1=self._get_new_color())
            shape = self.app.register_cube_shape(extents[0] * self.scaling, extents[1] * self.scaling, extents[2] * self.scaling, texture, 4)
            self._add_shape(shape, name)
            self._shape_geo_hash[geo_hash] = shape
        if not is_template:
            body = self._resolve_body(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_mesh(self, name: str, points, indices, colors=None, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0), update_topology=False, parent_body: str=None, is_template: bool=False):
        """Add a mesh for visualization
        
        Args:
            points: The points of the mesh
            indices: The indices of the mesh
            colors: The colors of the mesh
            name: A name for the USD prim on the stage
        """
        if name in self._mesh_name:
            self._update_mesh(name, points, indices, scale)
            self._update_instance(name, pos, rot)
            shape = self._mesh_name[name]
            return shape
        geo_hash = hash((int(warp.sim.GEO_MESH), tuple(np.array(points).flatten()), tuple(np.array(indices).flatten())))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self._update_instance(name, pos, rot):
                return shape
        else:
            if colors is None:
                color = (1.0, 0.8, 0.0)
            else:
                color = colors[0]
            texture = self.create_check_texture(color1=color, color2=color, width=1, height=1)
            faces = np.array(indices).reshape((-1, 3))
            vertices = np.array(points) * (np.array(scale)*self.scaling)
            shape = self._add_mesh(faces, vertices, (1.,1.,1.), texture)
            self._add_shape(shape, name)
            self._shape_geo_hash[geo_hash] = shape
            self._mesh_name[name] = shape
        if not is_template:
            body = self._resolve_body(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, scale)
        return shape

    def _render_lines(self, name: str, lines, color: tuple, radius: float=0.01):
        if len(lines) == 0:
            return
        if len(lines) > len(self._line_instance_ids[name]):
            if name not in self._line_shape:
                texture = self.create_check_texture(color1=color, color2=color, width=1, height=1)
                up_axis = 1
                half_height = 0.5
                shape = self.app.register_graphics_capsule_shape(radius * self.scaling, half_height * self.scaling, up_axis, texture)
                self._add_shape(shape, name)
                self._line_shape[name] = shape
            else:
                shape = self._line_shape[name]
            add_count = len(lines) - len(self._line_instance_ids[name])
            pos = [self.p.TinyVector3f(0., 0., 0.)] * add_count
            rot = [self.p.TinyQuaternionf(0., 0., 0., 1.)] * add_count
            color = [self.p.TinyVector3f(1., 1., 1.)] * add_count
            scale = [self.p.TinyVector3f(1., 1., 1.)] * add_count

            new_ids = self._add_instances(shape, pos, rot, color, scale)
            self._line_instance_ids[name].extend(new_ids)
            self._line_instance_ids_wp[name] = wp.array(self._line_instance_ids[name], dtype=wp.int32, device="cuda")
            
        # update lines
        vbo = self.app.cuda_map_vbo()
        vbo_positions = wp.array(
            ptr=vbo.positions, dtype=wp.vec4, shape=(self._instance_count,),
            device="cuda", owner=False, ndim=1)
        vbo_orientations = wp.array(
            ptr=vbo.orientations, dtype=wp.quat, shape=(self._instance_count,),
            device="cuda", owner=False, ndim=1)
        vbo_scalings = wp.array(
            ptr=vbo.scalings, dtype=wp.vec4, shape=(self._instance_count,),
            device="cuda", owner=False, ndim=1)
        lines_wp = wp.array(lines, dtype=wp.vec3, ndim=2, device="cuda")
        wp.launch(
            update_line_transforms,
            dim=len(lines),
            inputs=[
                self._line_instance_ids_wp[name],
                lines_wp,
                self.scaling,
            ],
            outputs=[
                vbo_positions,
                vbo_orientations,
                vbo_scalings,
            ],
            device="cuda")
        self.app.cuda_unmap_vbo()
    
    def render_line_list(self, name, vertices, indices, color, radius):
        """Add a line list as a set of capsules
        
        Args:
            vertices: The vertices of the line-list
            indices: The indices of the line-list
            color: The color of the line
            radius: The radius of the line
        """
        lines = []
        for i in range(len(indices)//2):
            lines.append((vertices[indices[2*i]], vertices[indices[2*i+1]]))
        lines = np.array(lines)
        self._render_lines(name, lines, color, radius)

    def render_line_strip(self, name: str, vertices, color: tuple, radius: float=0.01):
        """Add a line strip as a set of capsules
        
        Args:
            vertices: The vertices of the line-strip
            color: The color of the line
            radius: The radius of the line
        """
        lines = []
        for i in range(len(vertices)-1):
            lines.append((vertices[i], vertices[i+1]))
        lines = np.array(lines)
        self._render_lines(name, lines, color, radius)
    
    def render_points(self, name: str, points, radius, colors=None):
        """Add a set of points
        
        Args:
            points: The points to render
            radius: The radius of the points
            colors: The colors of the points
            name: A name for the USD prim on the stage
        """

        if len(points) == 0:
            return

        if isinstance(points, wp.array):
            points_wp = points
            if not points.device.is_cuda:
                points_wp = points.to(device="cuda")
        else:
            points_wp = wp.array(points, dtype=wp.vec3, device="cuda")

        if len(points) > len(self._point_instance_ids[name]):
            if name not in self._point_shape:
                texture = self.create_check_texture(color1=(1.,1.,1.), color2=(1.,1.,1.), width=1, height=1)
                shape = self.app.register_graphics_unit_sphere_shape(self.p.EnumSphereLevelOfDetail.SPHERE_LOD_LOW, texture)
                self._add_shape(shape, name)
                self._point_shape[name] = shape

            if isinstance(points, wp.array):
                points_np = points.numpy()
            else:
                points_np = np.array(points)
            
            add_count = len(points) - len(self._point_instance_ids[name])
            pos = [self.p.TinyVector3f(*p) for p in points_np[-add_count:]]
            rot = [self.p.TinyQuaternionf(0., 0., 0., 1.)] * add_count
            if colors is None:
                color = [self.p.TinyVector3f(*self.default_points_color)] * add_count
            else:
                color = [self.p.TinyVector3f(*c) for c in colors]
            scale = [self.p.TinyVector3f(radius*self.scaling, radius*self.scaling, radius*self.scaling)] * add_count
            new_ids = self._add_instances(self._point_shape[name], pos, rot, color, scale)
            self._point_instance_ids[name].extend(new_ids)
            self._point_instance_ids_wp[name] = wp.array(self._point_instance_ids[name], dtype=wp.int32, device="cuda")

        vbo = self.app.cuda_map_vbo()
        vbo_positions = wp.array(
            ptr=vbo.positions, dtype=wp.vec4, length=self._instance_count, device="cuda", owner=False, ndim=1)
        wp.launch(
            update_points_positions,
            dim=len(points),
            inputs=[self._point_instance_ids_wp[name], points_wp, self.scaling],
            outputs=[vbo_positions],
            device="cuda")
        self.app.cuda_unmap_vbo()
    
    def render_ref(self, name: str, path: str, pos: tuple, rot: tuple, scale: tuple):
        """
        Create a reference (instance) with the given name to the given path.
        """

        if path in self._body_name:
            # create a new body instance
            self._body_name[name] = len(self._body_name)

            body_id = self._body_name[path]
            for shape in self._body_shapes[body_id]:
                self.add_shape_instance(name, shape, pos, rot, scale)

            return

        if path in self._shape_name:
            # create a new shape instance
            shape = self._shape_name[path]
            self.add_shape_instance(name, shape, pos, rot, scale)

            return

        raise Exception("Cannot create reference to path: " + path)

    def update_body_transforms(self, body_tf: wp.array):
        body_q = None
        if body_tf is not None:
            if body_tf.device.is_cuda:
                body_q = body_tf
            else:
                body_q = body_tf.to("cuda")

        vbo = self.app.cuda_map_vbo()
        vbo_positions = wp.array(
            ptr=vbo.positions, dtype=wp.vec4, shape=(self._instance_count,),
            device="cuda", owner=False, ndim=1)
        vbo_orientations = wp.array(
            ptr=vbo.orientations, dtype=wp.quat, shape=(self._instance_count,),
            device="cuda", owner=False, ndim=1)
        wp.launch(
            update_vbo_transforms,
            dim=self._shape_count,
            inputs=[
                self._shape_instance_ids_wp,
                self._shape_body_wp,
                self._shape_transform_wp,
                body_q,
                self.scaling,
            ],
            outputs=[
                vbo_positions,
                vbo_orientations,
            ],
            device="cuda")
        self.app.cuda_unmap_vbo()

    def begin_frame(self, time: float):
        self.time = time
        if self.app.window.requested_exit():
            sys.exit(0)

    def end_frame(self):
        self.update()
        if self._has_new_instances:
            self.complete_setup()
            self.update_body_transforms(None)
        if len(self._shape_instance_updates):
            shape_transform = self._shape_transform_wp.numpy()
            shape_scale = self._shape_scale_wp.numpy()
            for idx, tf, scale in self._shape_instance_updates:
                shape_transform[idx] = tf
                if scale is not None:
                    shape_scale[idx] = scale
            self._shape_transform_wp = wp.array(shape_transform, dtype=wp.transform, device="cuda")
            self._shape_scale_wp = wp.array(shape_scale, dtype=wp.vec3, device="cuda")
            self._shape_instance_updates = []
            self.update_body_transforms(None)
        while self.paused and not self.app.window.requested_exit():
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
        # save just keeps the window open to allow the user to interact with the scene
        while not self.app.window.requested_exit():
            self.update()
        if self.app.window.requested_exit():
            sys.exit(0)

    def create_check_texture(self, color1=(0, 0.5, 1.0), color2=None, width=default_texture_size, height=default_texture_size):
        if width == 1 and height == 1:        
            pixels = np.array([np.array(color1)*255], dtype=np.uint8)
        else:
            pixels = np.zeros((width, height, 3), dtype=np.uint8)
            half_w = width // 2
            half_h = height // 2
            color1 = np.array(np.array(color1)*255, dtype=np.uint8)
            pixels[0:half_w, 0:half_h] = color1
            pixels[half_w:width, half_h:height] = color1
            if color2 is None:
                color2 = np.array(np.clip(np.array(color1, dtype=np.float32) + 50, 0, 255), dtype=np.uint8)
            else:
                color2 = np.array(np.array(color2)*255, dtype=np.uint8)
            pixels[half_w:width, 0:half_h] = color2
            pixels[0:half_w, half_h:height] = color2
        return self.app.renderer.register_texture(pixels.flatten().tolist(), width, height, False)
