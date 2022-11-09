# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This is the implementation of the OGN node defined in OgnParticleSolver.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import math

import numpy as np
import warp as wp
import warp.sim

import omni.timeline

from pxr import Usd, UsdGeom, Gf, Sdf

profile_enabled = False

# helper to read a USD xform out of graph inputs
def read_transform(input):
    xform = Gf.Matrix4d(input.reshape((4,4)))
    return xform

def read_transform_bundle(bundle):
    timeline =  omni.timeline.get_timeline_interface()
    time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.attribute_by_name("sourcePrimPath").value))
    return prim.ComputeLocalToWorldTransform(time)



def triangulate(counts, indices):

    # triangulate
    num_tris = np.sum(np.subtract(counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0
    for nb in counts:
        for i in range(nb-2):
            tri_indices[ctr] = indices[wedgeIdx]
            tri_indices[ctr + 1] = indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = indices[wedgeIdx + i + 2]
            ctr+=3
        wedgeIdx+=nb

    return tri_indices


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

    mesh_points[tid] = p
    mesh_velocities[tid] = v


class OgnParticleSolverState:

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.model = None
        self.state_0 = None
        self.state_1 = None
        self.mesh = None

        self.integrator = None
        
        # local space copy of particles positions (used when updating the output mesh)
        self.particles_positions_device = None
        self.particles_positions_host = None

        # local space copy of collider positions and velocities on the device
        self.collider_positions_current = None
        self.collider_positions_previous = None

        self.time = 0.0
        
        self.capture = None
        self.device = None

    # swap current / prev collider positions
    def swap(self):
        
        t = self.collider_positions_current
        self.collider_positions_current = self.collider_positions_previous
        self.collider_positions_previous = t

class OgnParticleSolver:

    @staticmethod
    def internal_state():

        return OgnParticleSolverState()

    """
    """
    @staticmethod
    def compute(db) -> bool:
        """Run simulation"""

        timeline =  omni.timeline.get_timeline_interface()
        state = db.internal_state
       
        if state.device == None:
            state.device = f"cuda:{db.inputs.device}"

        with wp.ScopedDevice(state.device):

            # reset on stop
            if (state.model and timeline.is_stopped()):
                state.reset()               

            time = timeline.get_current_time()

            # initialize
            if (timeline.is_playing() and state.time != time):
            
                state.time = time

                if state.model is None:
                    
                    # build particles
                    builder = wp.sim.ModelBuilder()

                    # particles
                    with wp.ScopedTimer("Create Particles", active=profile_enabled):

                        if (db.inputs.spawn_particles.valid):

                            particle_points = db.inputs.spawn_particles.attribute_by_name(db.tokens.points).value
                            particle_velocities = db.inputs.spawn_particles.attribute_by_name(db.tokens.velocities).value

                            # transform particles points to world-space
                            points_count = len(particle_points)
                            
                            for i in range(points_count):
                                builder.add_particle(pos=particle_points[i],
                                                     vel=particle_velocities[i],
                                                     mass=db.inputs.mass)


                    # collision shape
                    with wp.ScopedTimer("Create Collider"):

                        collider = db.inputs.collider

                        if (collider.valid):

                            collider_points = collider.attribute_by_name(db.tokens.points).value
                            collider_indices = collider.attribute_by_name(db.tokens.faceVertexIndices).value
                            collider_counts = collider.attribute_by_name(db.tokens.faceVertexCounts).value
                            collider_xform = read_transform_bundle(collider)

                            # save local copy
                            state.collider_positions_current = wp.array(collider_points, dtype=wp.vec3)
                            state.collider_positions_previous = wp.array(collider_points, dtype=wp.vec3)

                            world_positions = []
                            for i in range(len(collider_points)):
                                world_positions.append(collider_xform.Transform(Gf.Vec3f(tuple(collider_points[i]))))

                            state.mesh = wp.sim.Mesh(
                                world_positions,
                                triangulate(collider_counts, collider_indices),
                                compute_inertia=False)

                            builder.add_shape_mesh(
                                body=-1,
                                mesh=state.mesh,
                                pos=(0.0, 0.0, 0.0),
                                rot=(0.0, 0.0, 0.0, 1.0),
                                scale=(1.0, 1.0, 1.0))

                            state.collider_xform = read_transform_bundle(db.inputs.collider)

                    # finalize sim model
                    model = builder.finalize()
                    
                    # can only have one contact per-particle, since only one shape
                    model.allocate_soft_contacts(model.particle_count)

                    # create integrator
                    state.integrator = wp.sim.SemiImplicitIntegrator()
                    #state.integrator = wp.sim.VariationalImplicitIntegrator(model, solver="nesterov", max_iters=256, alpha=0.1, report=False)

                    # save model and state
                    state.model = model
                    state.state_0 = model.state()
                    state.state_1 = model.state()

                    state.positions_host = wp.zeros(model.particle_count, dtype=wp.vec3, device="cpu")
                    state.positions_device = wp.zeros(model.particle_count, dtype=wp.vec3)

                    
                # update dynamic properties
                state.model.ground = db.inputs.ground
                state.model.ground_plane = np.array((db.inputs.ground_plane[0], db.inputs.ground_plane[1], db.inputs.ground_plane[2], 0.0))

                state.model.gravity = db.inputs.gravity

                # contact properties
                state.model.particle_radius = db.inputs.radius
                state.model.particle_ke = db.inputs.k_contact_elastic
                state.model.particle_kd = db.inputs.k_contact_damp
                state.model.particle_kf = db.inputs.k_contact_friction
                state.model.particle_mu = db.inputs.k_contact_mu
                state.model.particle_cohesion = db.inputs.k_contact_cohesion
                state.model.particle_adhesion = db.inputs.k_contact_adhesion

                state.model.soft_contact_ke = db.inputs.k_contact_elastic
                state.model.soft_contact_kd = db.inputs.k_contact_damp
                state.model.soft_contact_kf = db.inputs.k_contact_friction
                state.model.soft_contact_mu = db.inputs.k_contact_mu
                state.model.soft_contact_distance = db.inputs.collider_offset
                state.model.soft_contact_margin = db.inputs.collider_offset*db.inputs.collider_margin
                

                # update collider positions
                with wp.ScopedTimer("Refit", active=profile_enabled):
                    
                    if (state.mesh):
                        
                        # swap prev/curr mesh positions
                        state.swap()

                        collider_points = db.inputs.collider.attribute_by_name(db.tokens.points).value
                        collider_points_host = wp.array(collider_points, dtype=wp.vec3, copy=False, device="cpu")

                        wp.copy(state.collider_positions_current, collider_points_host)

                        alpha = 0.0#(i+1)/sim_substeps

                        previous_xform = state.collider_xform
                        current_xform = read_transform_bundle(db.inputs.collider)

                        wp.launch(
                            kernel=transform_mesh, 
                            dim=len(state.mesh.vertices), 
                            inputs=[state.collider_positions_current,
                                    state.collider_positions_previous,
                                    np.array(current_xform).T,
                                    np.array(previous_xform).T,
                                    state.mesh.mesh.points,
                                    state.mesh.mesh.velocities,
                                    1.0/60.0,
                                    alpha])

                        state.collider_xform = current_xform

                        # refit bvh
                        state.mesh.mesh.refit()

                use_graph = True
                if (use_graph):
                    if (state.capture == None):

                        wp.capture_begin()

                        # simulate
                        sim_substeps = db.inputs.num_substeps
                        sim_dt = (1.0/60)/sim_substeps

                        # run collision detection once per-frame
                        wp.sim.collide(state.model, state.state_0)

                        for i in range(sim_substeps):

                            state.state_0.clear_forces()

                            state.integrator.simulate(
                                state.model, 
                                state.state_0, 
                                state.state_1, 
                                sim_dt)

                            (state.state_0, state.state_1) = (state.state_1, state.state_0)

                        state.capture = wp.capture_end()

                # step simulation
                with wp.ScopedTimer("Simulate", active=profile_enabled):
                    
                    if (use_graph):

                        state.model.particle_grid.build(state.state_0.particle_q, db.inputs.radius + db.inputs.particle_margin)
                        wp.capture_launch(state.capture)
                    else:
                        
                        # simulate
                        sim_substeps = db.inputs.num_substeps
                        sim_dt = (1.0/60)/sim_substeps

                        # run collision detection once per-frame
                        wp.sim.collide(state.model, state.state_0)

                        state.model.particle_grid.build(state.state_0.particle_q, db.inputs.radius + db.inputs.particle_margin)

                        for i in range(sim_substeps):

                            state.state_0.clear_forces()

                            state.integrator.simulate(
                                state.model,  
                                state.state_0, 
                                state.state_1, 
                                sim_dt)

                            (state.state_0, state.state_1) = (state.state_1, state.state_0)

                with wp.ScopedTimer("Synchronize", active=profile_enabled):

                    # back to host for OG outputs
                    wp.copy(state.positions_host, state.state_0.particle_q)

                    if db.inputs.asynchronous == False:
                        wp.synchronize()

                with wp.ScopedTimer("Write", active=profile_enabled):

                    db.outputs.positions_size = len(state.positions_host)
                    db.outputs.positions[:] = state.positions_host.numpy()

            else:
                with wp.ScopedTimer("Write", active=profile_enabled):
                    
                    # timeline not playing and sim. not yet initialized, just pass through outputs
                    if state.model is None and db.inputs.spawn_particles.valid:

                        particle_points = db.inputs.spawn_particles.attribute_by_name(db.tokens.points).value
                        particle_velocities = db.inputs.spawn_particles.attribute_by_name(db.tokens.velocities).value

                        db.outputs.positions_size = len(particle_points)
                        db.outputs.positions[:] = particle_points

                        db.outputs.widths_size = len(particle_points)
                        db.outputs.widths[:] = np.ones(len(particle_points))*db.inputs.radius*2.0

                        # db.outputs.ids_size = len(particle_points)
                        # db.outputs.ids[:] = [0]*len(particle_points)


        return True
