import math
import os
import sys

# to allow tests to import the module they belong to
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np

import tests.test_sim_util as util
from tests.render_sim import SimRenderer

wp.init()

class HumanoidSNU:

    name = "humanoid_snu_lower" 

    initial_y = 1.0

    def __init__(self, render=True, sim_duration=1.0, device='cpu'):

        self.frame_dt = 1.0/60.0

        self.episode_duration = 2.0      # seconds
        self.episode_frames = int(self.episode_duration/self.frame_dt)

        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = int(self.episode_duration / self.sim_dt)
        self.sim_time = 0.0
        self.render_time = 0.0

        np.random.seed(41)

        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.filter = {}

        if self.name == "humanoid_snu_arm":
            self.filter = { "ShoulderR", "ArmR", "ForeArmR", "HandR", "Torso", "Neck" }
            self.ground = False

        if self.name == "humanoid_snu_neck":
            self.filter = { "Torso", "Neck", "Head", "ShoulderR", "ShoulderL" }
            self.ground = False

        if self.name == "humanoid_snu_lower":
            self.filter = { "Pelvis", "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR", "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL"}
            self.ground = True
            self.initial_y = 1.0

        if self.name == "humanoid_snu":
            self.filter = {}
            self.ground = True


        self.skeletons = []

        for i in range(10):
            
            skeleton = util.Skeleton(wp.transform((i*2.0, 0.0, 0.0), wp.quat_identity()), "./tests/assets/snu/arm.xml", "./tests/assets/snu/muscle284.xml", builder, self.filter, armature=0.05)

            self.skeletons.append(skeleton)

        # builder.joint_limit_ke *= 0.1
        # builder.joint_limit_kd *= 0.1
        # builder.joint_target_ke *= 0.1
        # builder.joint_target_kd *= 0.1

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = self.ground
        self.model.joint_attach_ke *= 8.0
        self.model.joint_attach_kd *= 0.2

        self.integrator = wp.sim.SemiImplicitIntegrator()

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = SimRenderer(self.model, "./tests/outputs/" + self.name + ".usd")


    def run(self):



        self.sim_time = 0.0
        self.state = self.model.state()

        joint_q = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        joint_qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        # set initial position 1m off the ground
        for i in range(len(self.skeletons)):
            joint_q[self.skeletons[i].coord_start + 0] = i*1.5
            joint_q[self.skeletons[i].coord_start + 1] = self.initial_y
        
        if (self.model.ground):
            self.model.collide(self.state)


        # create update graph
        wp.capture_begin()

        # simulate
        for i in range(0, self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            self.sim_time += self.sim_dt
                
        graph = wp.capture_end()

        with wp.ScopedTimer("simulate", detailed=False, active=True):

            for f in range(0, self.episode_frames):

                wp.capture_launch(graph)
                self.sim_time += self.sim_dt

                # # # simulate
                # for i in range(0, self.sim_substeps):
                #     self.state.clear_forces()
                #     self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                #     self.sim_time += self.sim_dt

#            wp.synchronize()      

                # render
                with wp.ScopedTimer("render", True):

                    if (self.render):
                        self.render_time += self.frame_dt
                        
                        self.renderer.begin_frame(self.render_time)
                        self.renderer.render(self.state)
                        self.renderer.end_frame()

        self.renderer.save()

env = HumanoidSNU(render=True, sim_duration=2.0, device='cuda')
env.run()

