import torch
import time
import math

# include parent path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import warp.sim
import warp.render
import warp.torch

wp.init()

from pxr import Usd, UsdGeom, Gf

#---------------------------------

class Ballistic:

    sim_duration = 0.5       # seconds

    # control frequency
    frame_dt = 1.0/60.0
    frame_steps = int(sim_duration/frame_dt)

    # sim frequency
    sim_substeps = 2
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_time = 0.0

    render_time = 0.0

    train_iters = 500
    train_rate = 100.0         #1.0/(sim_dt*sim_dt)

    def __init__(self, render=True, adapter='cpu'):

        builder = wp.sim.ModelBuilder()

        builder.add_particle(pos=(0, 1.0, 0.0), vel=(0.1, 0.0, 0.0), mass=1.0)

        self.device = adapter
        self.render = render

        self.model = builder.finalize(adapter)
        self.model.ground = False

        self.target = torch.tensor((2.0, 1.0, 0.0), device=adapter)
        self.control = torch.zeros((self.frame_steps, 3), dtype=torch.float32, device=adapter, requires_grad=True)

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if (self.render):
            self.stage = wp.render.UsdRenderer("tests/outputs/test_sim_grad_ballistic.usda")

        # allocate input/output states
        self.states = []
        for i in range(self.sim_substeps+1):
            self.states.append(self.model.state(requires_grad=True))

    def step(self):
        for i in range(self.sim_substeps):

            self.states[i].clear_forces()
            self.states[i+1].clear_forces()

            self.integrator.simulate(self.model, self.states[i], self.states[i+1], self.sim_dt)


    # define PyTorch autograd op to wrap simulate func
    class StepFunc(torch.autograd.Function):

        # inputs should be anything you want gradients for
        # anything else will be treated as a constant and 
        # accessed directly through the env variable
        @staticmethod
        def forward(ctx, env, particle_q, particle_qd, particle_f):

            ctx.env = env
            ctx.particle_q = particle_q
            ctx.particle_qd = particle_qd
            ctx.particle_f = particle_f

            # apply initial conditions
            wp.copy(env.states[0].particle_q, wp.torch.from_torch(ctx.particle_q, dtype=wp.vec3))
            wp.copy(env.states[0].particle_qd, wp.torch.from_torch(ctx.particle_qd, dtype=wp.vec3))
            wp.copy(env.states[0].particle_f, wp.torch.from_torch(ctx.particle_f, dtype=wp.vec3))

            # simulate
            ctx.env.step()

            # copy output back to torch
            out_q = wp.torch.to_torch(env.states[-1].particle_q)
            out_qd = wp.torch.to_torch(env.states[-1].particle_qd)
            
            # need to be careful not to keep a reference to output tensors since otherwise it creates a GC cycle
            # where outputs reference this node, and this node references outputs
            return (out_q, out_qd)

        @staticmethod
        def backward(ctx, *grads):

            # re-apply initial conditions
            wp.copy(ctx.env.states[0].particle_q, wp.torch.from_torch(ctx.particle_q, dtype=wp.vec3))
            wp.copy(ctx.env.states[0].particle_qd, wp.torch.from_torch(ctx.particle_qd, dtype=wp.vec3))
            wp.copy(ctx.env.states[0].particle_f, wp.torch.from_torch(ctx.particle_f, dtype=wp.vec3))

            # re-simulate
            tape = wp.Tape()
            with tape:
                ctx.env.step()

            # match up the torch grads to state grads
            adj_user = { ctx.env.states[-1].particle_q: wp.torch.from_torch(grads[0], dtype=wp.vec3),
                         ctx.env.states[-1].particle_qd: wp.torch.from_torch(grads[1], dtype=wp.vec3) }

            # back-prop
            tape.backward(grads=adj_user)
 
            # copy grad back to Torch
            grads = (None, wp.torch.to_torch(tape.gradients[ctx.env.states[0].particle_q]),
                           wp.torch.to_torch(tape.gradients[ctx.env.states[0].particle_qd]),
                           wp.torch.to_torch(tape.gradients[ctx.env.states[0].particle_f]))

            return grads



    def loss(self):

        # initial state
        particle_q = wp.torch.to_torch(self.model.particle_q)
        particle_qd = wp.torch.to_torch(self.model.particle_qd)

        # run control loop
        for f in range(self.frame_steps):
            (particle_q, particle_qd) = Ballistic.StepFunc.apply(self, particle_q, particle_qd, self.control[f])

            if (self.render):
                self.stage.begin_frame(self.render_time)
                self.stage.render_sphere(particle_q.tolist()[0], rot=wp.quat_identity(), radius=0.1, name="ball")
                self.stage.render_sphere(self.target.tolist(), rot=wp.quat_identity(), radius=0.1, name="target")
                self.stage.end_frame()

            self.render_time += self.frame_dt

        if (self.render):
            self.stage.save()

        # compute final loss
        pos_penalty = 1.0
        vel_penalty = 0.1

        loss = torch.norm(particle_q[0] - self.target)*pos_penalty + torch.norm(particle_qd)*vel_penalty
        return loss

    def train(self, mode='gd'):

        params = [self.control]

        # Gradient Descent
        if (mode == 'gd'):
            for i in range(self.train_iters):
                
                with wp.ScopedTimer("forward", active=False):
                    l = self.loss()

                with wp.ScopedTimer("backward", active=False):
                    l.backward()

                print(l)

                with torch.no_grad():
                    for p in params:
                        p -= self.train_rate * p.grad
                        p.grad.zero_()

        # L-BFGS
        if (mode == 'lbfgs'):

            optimizer = torch.optim.LBFGS(params, 1.0, tolerance_grad=1.e-5, history_size=4, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                l = self.loss()
                l.backward()

                print(f"Loss: {l.item()}")

                return l

            for i in range(self.train_iters):
                optimizer.step(closure)

        # SGD
        if (mode == 'sgd'):

            optimizer = torch.optim.SGD(params, lr=self.train_rate, momentum=0.8)

            for i in range(self.train_iters):
                optimizer.zero_grad()

                l = self.loss()
                l.backward()

                print(f"Loss: {l.item()}")

                optimizer.step()


    def check_grad(self):

        particle_q = wp.torch.to_torch(self.model.particle_q)
        particle_qd = wp.torch.to_torch(self.model.particle_qd)

        particle_q.requires_grad = True
        particle_qd.requires_grad = True

        torch.autograd.gradcheck(Ballistic.StepFunc.apply, (self, particle_q, particle_qd, self.control[0]), eps=1e-3, atol=1e-3, rtol=1.e-3, raise_exception=True)#, nondet_tol=1.e+6)

#---------

ballistic = Ballistic(adapter="cpu", render=True)

#ballistic.check_grad()
ballistic.train('gd')
