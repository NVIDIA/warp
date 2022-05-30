# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pxr import Usd, UsdGeom, Gf, Sdf

import math

import warp as wp
import warp.sim

class SimRenderer(warp.render.UsdRenderer):
    
    def __init__(self, model: warp.sim.Model, path):

        # create USD stage
        super().__init__(path)

        self.model = model

        # add ground plane
        if (self.model.ground):
            self.render_ground(size=20.0)

        # create rigid body root node
        for b in range(model.body_count):
            xform = UsdGeom.Xform.Define(self.stage, self.root.GetPath().AppendChild("body_" + str(b)))
            wp.render._usd_add_xform(xform)

        # create rigid shape children
        if (self.model.shape_count):
            shape_body = model.shape_body.numpy()
            shape_geo_src = model.shape_geo_src#.numpy()
            shape_geo_type = model.shape_geo_type.numpy()
            shape_geo_scale = model.shape_geo_scale.numpy()
            shape_transform = model.shape_transform.numpy()

            for s in range(model.shape_count):
            
                parent_path = self.root.GetPath()
                if shape_body[s] >= 0:
                    parent_path = parent_path.AppendChild("body_" + str(shape_body[s].item()))

                geo_type = shape_geo_type[s]
                geo_scale = shape_geo_scale[s]
                geo_src = shape_geo_src[s]

                # shape transform in body frame
                X_bs = warp.transform_expand(shape_transform[s])

                if (geo_type == warp.sim.GEO_PLANE):

                    # plane mesh
                    size = 1000.0

                    mesh = UsdGeom.Mesh.Define(self.stage, parent_path.AppendChild("plane_" + str(s)))
                    mesh.CreateDoubleSidedAttr().Set(True)

                    points = ((-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size))
                    normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
                    counts = (4, )
                    indices = [0, 1, 2, 3]

                    mesh.GetPointsAttr().Set(points)
                    mesh.GetNormalsAttr().Set(normals)
                    mesh.GetFaceVertexCountsAttr().Set(counts)
                    mesh.GetFaceVertexIndicesAttr().Set(indices)

                elif (geo_type == warp.sim.GEO_SPHERE):

                    mesh = UsdGeom.Sphere.Define(self.stage, parent_path.AppendChild("sphere_" + str(s)))
                    mesh.GetRadiusAttr().Set(float(geo_scale[0]))

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bs.p, X_bs.q, (1.0, 1.0, 1.0), 0.0)

                elif (geo_type == warp.sim.GEO_CAPSULE):
                    mesh = UsdGeom.Capsule.Define(self.stage, parent_path.AppendChild("capsule_" + str(s)))
                    mesh.GetRadiusAttr().Set(float(geo_scale[0]))
                    mesh.GetHeightAttr().Set(float(geo_scale[1] * 2.0))

                    # geometry transform w.r.t shape, convert USD geometry to physics engine convention
                    X_sg = warp.transform((0.0, 0.0, 0.0), warp.utils.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 0.5))
                    X_bg = warp.utils.transform_multiply(X_bs, X_sg)

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bg.p, X_bg.q, (1.0, 1.0, 1.0), 0.0)

                elif (geo_type == warp.sim.GEO_BOX):
                    mesh = UsdGeom.Cube.Define(self.stage, parent_path.AppendChild("box_" + str(s)))
                    #mesh.GetSizeAttr().Set((geo_scale[0], geo_scale[1], geo_scale[2]))

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bs.p, X_bs.q, (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

                elif (geo_type == warp.sim.GEO_MESH):

                    mesh = UsdGeom.Mesh.Define(self.stage, parent_path.AppendChild("mesh_" + str(s)))
                    mesh.GetPointsAttr().Set(geo_src.vertices)
                    mesh.GetFaceVertexIndicesAttr().Set(geo_src.indices)
                    mesh.GetFaceVertexCountsAttr().Set([3] * int(len(geo_src.indices) / 3))

                    wp.render._usd_add_xform(mesh)
                    wp.render._usd_set_xform(mesh, X_bs.p, X_bs.q, (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

                elif (geo_type == warp.sim.GEO_SDF):
                    pass
        


    def render(self, state: warp.sim.State):


        if (self.model.particle_count):

            particle_q = state.particle_q.numpy()

            # render particles
            self.render_points("particles", particle_q, radius=0.1)

            # render tris
            if (self.model.tri_count):
                self.render_mesh("surface", particle_q, self.model.tri_indices.numpy().flatten())

            # render springs
            if (self.model.spring_count):
                self.render_line_list("springs", particle_q, self.model.spring_indices.numpy().flatten(), [], 0.1)

        # render muscles
        if (self.model.muscle_count):
            
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

                    points.append(Gf.Vec3f(wp.transform_point(X_sc, point).tolist()))
                
                self.render_line_strip(name=f"muscle_{m}", vertices=points, radius=0.0075, color=(muscle_activation[m], 0.2, 0.5))
        
        
        with Sdf.ChangeBlock():

            # update  bodies
            if (self.model.body_count):

                body_q = state.body_q.numpy()

                for b in range(self.model.body_count):

                    node = UsdGeom.Xform(self.stage.GetPrimAtPath(self.root.GetPath().AppendChild("body_" + str(b))))

                    # unpack rigid transform
                    X_sb = warp.transform_expand(body_q[b])

                    wp.render._usd_set_xform(node, X_sb.p, X_sb.q, (1.0, 1.0, 1.0), self.time)


