# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This is the implementation of the OGN node defined in OgnCloth.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import math

import numpy as np
import warp as wp
import warp.sim

import omni.timeline
import omni.warp

from pxr import Usd, UsdGeom, Gf, Sdf


# helper to get the transform from a bundle prim
def read_transform_bundle(bundle):

    timeline =  omni.timeline.get_timeline_interface()
    time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()

    stage = omni.usd.get_context().get_stage()
    prim_path = bundle.attribute_by_name("sourcePrimPath").value
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))

    return prim.ComputeLocalToWorldTransform(time)

# helper to read points from a bundle prim
def read_points_bundle(bundle):
    return bundle.attribute_by_name("points").value

# helper to read indices from a bundle prim
def read_indices_bundle(bundle):
    return bundle.attribute_by_name("faceVertexIndices").value

# transform points from local space to world space given a mat44
@wp.kernel
def transform_points(src: wp.array(dtype=wp.vec3),
                     dest: wp.array(dtype=wp.vec3),
                     xform: wp.mat44):

    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m

@wp.kernel
def update_tri_materials(dest: wp.array2d(dtype=float),tri_ke:float, tri_ka:float, tri_kd:float, tri_lift:float, tri_drag:float):
    tid = wp.tid()
    dest[tid,0]=tri_ke
    dest[tid,1]=tri_ka
    dest[tid,2]=tri_kd
    dest[tid,3]=tri_drag
    dest[tid,4]=tri_lift

@wp.kernel
def update_edge_properties(dest: wp.array2d(dtype=float), edge_ke:float, edge_kd:float):
    tid = wp.tid()
    dest[tid,0]=edge_ke
    dest[tid,1]=edge_kd
    

# update mesh data given two sets of collider positions
# computes velocities and transforms points to world-space
@wp.kernel
def transform_mesh(collider_current: wp.array(dtype=wp.vec3),
                   collider_previous: wp.array(dtype=wp.vec3),
                   xform_current: wp.mat44,
                   xform_previous: wp.mat44,
                   mesh_points: wp.array(dtype=wp.vec3),
                   mesh_velocities: wp.array(dtype=wp.vec3),
                   dt: float,
                   alpha: float):

    tid = wp.tid()

    local_p1 = collider_current[tid]
    local_p0 = collider_previous[tid]

    world_p1 = wp.transform_point(xform_current, local_p1)
    world_p0 = wp.transform_point(xform_previous, local_p0)

    p = world_p1*alpha + world_p0*(1.0-alpha)
    v = (world_p1-world_p0)/dt

    mesh_points[tid]= p
    mesh_velocities[tid] = v


class OgnClothState:

    def __init__(self):
        self.reset()

    def reset(self):
        self.model = None
        self.state_0 = None
        self.state_1 = None
        self.mesh = None

        self.integrator = None
        
        # local space copy of collider positions and velocities on the device
        self.collider_positions_current = None
        self.collider_positions_previous = None

        self.time = 0.0
        
        self.capture = None

    # swap current / prev collider positions
    def swap(self):
        
        t = self.collider_positions_current
        self.collider_positions_current = self.collider_positions_previous
        self.collider_positions_previous = t

class OgnCloth:

    @staticmethod
    def internal_state():

        return OgnClothState()

    """
    """
    @staticmethod
    def compute(db) -> bool:
        """Run simulation"""

        timeline =  omni.timeline.get_timeline_interface()
        context = db.internal_state

        with wp.ScopedDevice("cuda:0"):

            # reset on stop
            if (timeline.is_stopped()):
                context.reset()               

            # initialization
            if (timeline.is_playing()):
            
                if context.model is None:
                    
                    # build cloth
                    builder = wp.sim.ModelBuilder()

                    # cloth
                    with wp.ScopedTimer("Create Cloth", detailed=False):

                        if (db.inputs.cloth.valid):

                            # transform cloth points to world-space
                            cloth_xform = read_transform_bundle(db.inputs.cloth)
                            cloth_positions = read_points_bundle(db.inputs.cloth)
                            cloth_indices = read_indices_bundle(db.inputs.cloth)

                            density = db.inputs.density

                            # transform particles to world space
                            world_positions = []
                            for i in range(len(cloth_positions)):
                                world_positions.append(cloth_xform.Transform(Gf.Vec3f(tuple(cloth_positions[i]))))
                        
                            builder.add_cloth_mesh(pos=(0.0, 0.0, 0.0),
                                                   rot=(0.0, 0.0, 0.0, 1.0),
                                                   scale=1.0,
                                                   vel=(0.0, 0.0, 0.0),
                                                   vertices=world_positions,
                                                   indices=cloth_indices,
                                                   density=density,
                                                   tri_ke=db.inputs.k_tri_elastic,
                                                   tri_ka=db.inputs.k_tri_area,
                                                   tri_kd=db.inputs.k_tri_damp,
                                                   edge_ke=db.inputs.k_edge_bend,
                                                   edge_kd=db.inputs.k_edge_damp
                                                   )


                            avg_mass = np.mean(builder.particle_mass)

                            # set uniform mass to average mass to avoid large mass ratios
                            builder.particle_mass = np.array([avg_mass]*len(builder.particle_mass))


                    # collision shape
                    with wp.ScopedTimer("Create Collider"):

                        if (db.inputs.collider.valid):

                            collider_xform = read_transform_bundle(db.inputs.collider)
                            collider_positions = read_points_bundle(db.inputs.collider)
                            collider_indices = read_indices_bundle(db.inputs.collider)

                            # save local copy
                            context.collider_positions_current = wp.array(collider_positions, dtype=wp.vec3)
                            context.collider_positions_previous = wp.array(collider_positions, dtype=wp.vec3)

                            world_positions = []
                            for i in range(len(collider_positions)):
                                world_positions.append(collider_xform.Transform(Gf.Vec3f(tuple(collider_positions[i]))))

                            context.mesh = wp.sim.Mesh(
                                world_positions,
                                collider_indices,
                                compute_inertia=False)

                            builder.add_shape_mesh(
                                body=-1,
                                mesh=context.mesh,
                                pos=(0.0, 0.0, 0.0),
                                rot=(0.0, 0.0, 0.0, 1.0),
                                scale=(1.0, 1.0, 1.0))

                    # finalize sim model
                    model = builder.finalize()
                    
                    # can only have one contact per-particle, since only one shape
                    model.allocate_soft_contacts(model.particle_count)

                    # create integrator
                    context.integrator = wp.sim.SemiImplicitIntegrator()

                    # save model and state
                    context.model = model
                    context.state_0 = model.state()
                    context.state_1 = model.state()

                    context.positions_device = wp.zeros(model.particle_count, dtype=wp.vec3)

                    context.collider_xform = read_transform_bundle(db.inputs.collider)

                    # store stretch properties in model
                    context.model.tri_ke = db.inputs.k_tri_elastic
                    context.model.tri_ka = db.inputs.k_tri_area
                    context.model.tri_kd = db.inputs.k_tri_damp  

                    # store bending properties in model
                    context.model.edge_ke = db.inputs.k_edge_bend
                    context.model.edge_kd = db.inputs.k_edge_damp 


                # update dynamic properties
                context.model.ground = db.inputs.ground
                context.model.ground_plane = np.array((db.inputs.ground_plane[0], db.inputs.ground_plane[1], db.inputs.ground_plane[2], 0.0))

                # stretch properties
                context.model.gravity = db.inputs.gravity
                
                # contact properties
                context.model.soft_contact_ke = db.inputs.k_contact_elastic
                context.model.soft_contact_kd = db.inputs.k_contact_damp
                context.model.soft_contact_kf = db.inputs.k_contact_friction
                context.model.soft_contact_mu = db.inputs.k_contact_mu
                context.model.soft_contact_distance = db.inputs.collider_offset
                context.model.soft_contact_margin = db.inputs.collider_offset*10.0

                # Update tri properties if required:
                if(db.inputs.k_tri_elastic != context.model.tri_ke or
                    db.inputs.k_tri_area != context.model.tri_ka or
                    db.inputs.k_tri_damp != context.model.tri_kd):
                    wp.launch(update_tri_materials, 
                        dim=context.model.tri_count,
                        inputs=[context.model.tri_materials, db.inputs.k_tri_elastic, db.inputs.k_tri_area, db.inputs.k_tri_damp,0.0,0.0]
                    )

                # Update bending properties if required:
                if(db.inputs.k_edge_bend != context.model.edge_ke or db.inputs.k_edge_damp != context.model.edge_kd):
                    wp.launch(update_edge_properties, 
                        dim=context.model.edge_count,
                        inputs=[context.model.edge_bending_properties, db.inputs.k_edge_bend, db.inputs.k_edge_damp]
                    )

                # update collider positions
                with wp.ScopedTimer("Refit", active=False):
                    
                    if (context.mesh):
                        
                        # swap prev/curr mesh positions
                        context.swap()

                        # update current
                        collider_points_host = wp.array(read_points_bundle(db.inputs.collider), dtype=wp.vec3, copy=False, device="cpu")
                        wp.copy(context.collider_positions_current, collider_points_host)

                        alpha = 1.0#(i+1)/sim_substeps

                        previous_xform = context.collider_xform
                        current_xform = read_transform_bundle(db.inputs.collider)

                        wp.launch(
                            kernel=transform_mesh, 
                            dim=len(context.mesh.vertices), 
                            inputs=[context.collider_positions_current,
                                    context.collider_positions_previous,
                                    np.array(current_xform).T,
                                    np.array(previous_xform).T,
                                    context.mesh.mesh.points,
                                    context.mesh.mesh.velocities,
                                    1.0/60.0,
                                    alpha])

                        context.collider_xform = current_xform

                        # refit bvh
                        context.mesh.mesh.refit()

                use_graph = True
                if (use_graph):
                    if (context.capture == None):
                        
                        wp.capture_begin()

                        # simulate
                        sim_substeps = db.inputs.num_substeps
                        sim_dt = (1.0/60)/sim_substeps

                        # run collision detection once per-frame
                        wp.sim.collide(context.model, context.state_0)

                        for i in range(sim_substeps):

                            context.state_0.clear_forces()

                            context.integrator.simulate(
                                context.model, 
                                context.state_0, 
                                context.state_1, 
                                sim_dt)

                            (context.state_0, context.state_1) = (context.state_1, context.state_0)

                        context.capture = wp.capture_end()

                # step simulation
                with wp.ScopedTimer("Simulate", active=False):
                    
                    if (use_graph):
                        wp.capture_launch(context.capture)
                    else:
                        
                        # simulate
                        sim_substeps = db.inputs.num_substeps
                        sim_dt = (1.0/60)/sim_substeps

                        # run collision detection once per-frame
                        wp.sim.collide(context.model, context.state_0)

                        for i in range(sim_substeps):

                            context.state_0.clear_forces()

                            context.integrator.simulate(
                                context.model,  
                                context.state_0, 
                                context.state_1, 
                                sim_dt)

                            (context.state_0, context.state_1) = (context.state_1, context.state_0)


                # transform cloth positions back to local space
                with wp.ScopedTimer("Transform", active=False):
                    
                    cloth_xform_inv = read_transform_bundle(db.inputs.cloth).GetInverse()

                    wp.launch(kernel=transform_points, 
                              dim=context.model.particle_count, 
                              inputs=[context.state_0.particle_q, 
                                      context.positions_device, 
                                      np.array(cloth_xform_inv).T])

                with wp.ScopedTimer("Write", active=False):

                    db.outputs.positions_size = len(context.positions_device)

                    # copy points 
                    points_out = omni.warp.from_omni_graph(db.outputs.positions, dtype=wp.vec3)
                    wp.copy(points_out, context.positions_device)


        return True
