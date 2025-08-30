import argparse

import jax
import jax.numpy as jp
import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.jax_experimental.ffi import jax_ad_kernel
from warp.sim.articulation import eval_articulation_fk as fk_kernel

MASK_DTYPE = jp.bool_


class Example:
    def __init__(self, stage_path="example_inverse_kinematics_jax.usd", verbose=False):
        self.verbose = verbose

        fps = 60
        self.frame_dt = 1.0 / fps
        self.render_time = 0.0

        builder = wp.sim.ModelBuilder()
        builder.add_articulation()

        chain_length = 4
        chain_width = 1.0

        for i in range(chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 0.0, 0.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform([chain_width, 0.0, 0.0], wp.quat_identity())

            b = builder.add_body(origin=wp.transform([i, 0.0, 0.0], wp.quat_identity()), armature=0.1)

            builder.add_joint_revolute(
                parent=parent,
                child=b,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=parent_joint_xform,
                child_xform=wp.transform_identity(),
                limit_lower=-np.deg2rad(60.0),
                limit_upper=np.deg2rad(60.0),
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=30.0,
                limit_kd=30.0,
            )

            if i == chain_length - 1:
                builder.add_shape_sphere(pos=wp.vec3(0.0, 0.0, 0.0), radius=0.1, density=10.0, body=b)
            else:
                builder.add_shape_box(
                    pos=wp.vec3(chain_width * 0.5, 0.0, 0.0), hx=chain_width * 0.5, hy=0.1, hz=0.1, density=10.0, body=b
                )

        self.model = builder.finalize()
        self.model.ground = False

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=50.0)
        else:
            self.renderer = None

        self.target = jp.array(np.array((2.0, 1.0, 0.0), dtype=np.float32))

        self.body_q = None
        self.body_qd = None

        self.joint_q = jp.zeros((len(self.model.joint_q),), dtype=jp.float32)
        self.joint_qd = jp.zeros((len(self.model.joint_qd),), dtype=jp.float32)

        self.train_rate = 0.01

        # Wrap a Warp kernel with JAX AD support
        # - We use jax_ad_kernel to make the Warp kernel `eval_articulation_fk` callable from JAX
        #   and differentiable (custom_vjp under the hood via the XLA FFI path).
        # - launch_dim_arg_index=1 means the kernel will launch one thread per element of
        #   the second argument (the articulation mask), matching Warp's per-articulation launch.
        # - output_dims=(body_count,) tells JAX the forward output shapes, because the outputs are
        #   sized by the model and not by input arrays, which helps shape inference and the VJP.
        # - Gradients are produced by launching the same kernel in adjoint mode on the backward pass.
        self._jax_fk = jax_ad_kernel(
            fk_kernel,
            num_outputs=2,
            launch_dim_arg_index=1,
            output_dims=(self.model.body_count,),
        )

        m = self.model
        # Convert persistent model buffers to JAX arrays once (no copies)
        self._jax_articulation_start = wp.to_jax(m.articulation_start)
        # Boolean mask enables FK per articulation; also used to set launch dims
        self._jax_mask = jp.ones((m.articulation_count,), dtype=MASK_DTYPE)
        self._jax_joint_q_start = wp.to_jax(m.joint_q_start)
        self._jax_joint_qd_start = wp.to_jax(m.joint_qd_start)
        self._jax_joint_type = wp.to_jax(m.joint_type)
        self._jax_joint_parent = wp.to_jax(m.joint_parent)
        self._jax_joint_child = wp.to_jax(m.joint_child)
        self._jax_joint_X_p = wp.to_jax(m.joint_X_p)
        self._jax_joint_X_c = wp.to_jax(m.joint_X_c)
        self._jax_joint_axis = wp.to_jax(m.joint_axis)
        self._jax_joint_axis_start = wp.to_jax(m.joint_axis_start)
        self._jax_joint_axis_dim = wp.to_jax(m.joint_axis_dim)
        self._jax_body_com = wp.to_jax(m.body_com)

        # JIT-compiled forward kinematics under JAX
        # Inputs are JAX arrays (joint_q, joint_qd). The wrapped Warp kernel runs
        # on the active CUDA stream via FFI, returns JAX arrays (body transforms, body velocities).
        def fk_call(jq, jqd):
            return self._jax_fk(
                self._jax_articulation_start,
                self._jax_mask,
                jq,
                jqd,
                self._jax_joint_q_start,
                self._jax_joint_qd_start,
                self._jax_joint_type,
                self._jax_joint_parent,
                self._jax_joint_child,
                self._jax_joint_X_p,
                self._jax_joint_X_c,
                self._jax_joint_axis,
                self._jax_joint_axis_start,
                self._jax_joint_axis_dim,
                self._jax_body_com,
            )

        # Use jax.jit to compile the end-to-end FK call
        self._fk_call = jax.jit(fk_call)

    def forward(self):
        # Run FK under JAX and keep the JAX outputs for loss computation
        body_q, body_qd = self._fk_call(self.joint_q, self.joint_qd)
        self.body_q = wp.from_jax(body_q, dtype=wp.transform)
        self.body_qd = wp.from_jax(body_qd, dtype=wp.spatial_vector)
        pos = body_q[self.model.body_count - 1, 0:3]
        diff = pos - self.target
        self.loss = jp.sum(diff * diff)
        return self.loss

    def step(self):
        # Define a pure-JAX loss: run FK with q, compare end-effector position to target
        def loss_from_fk_outputs(body_q, body_qd):
            return jp.sum((body_q[self.model.body_count - 1, 0:3] - self.target) ** 2.0)

        def loss_fn(q):
            body_q, body_qd = self._fk_call(q, self.joint_qd)
            return loss_from_fk_outputs(body_q, body_qd)

        # Differentiate w.r.t. joint_q. The VJP calls Warp adjoint through the FFI-backed wrapper.
        grad_fn = jax.jit(jax.grad(loss_fn))
        g = grad_fn(self.joint_q)
        self.joint_q = self.joint_q - self.train_rate * g
        current_loss = self.forward()
        if self.verbose:
            print(f"loss: {float(current_loss)}")

    def render(self):
        if self.renderer is None:
            return
        s = self.model.state()
        s.body_q = self.body_q
        s.body_qd = self.body_qd
        self.renderer.begin_frame(self.render_time)
        self.renderer.render(s)
        self.renderer.render_sphere(
            name="target", pos=self.target, rot=wp.quat_identity(), radius=0.1, color=(1.0, 0.0, 0.0)
        )
        self.renderer.end_frame()
        self.render_time += self.frame_dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_inverse_kinematics_jax.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--train_iters", type=int, default=512, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    os_env = {}
    try:
        import os as _os

        os_env = _os.environ
        os_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os_env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    except Exception:
        pass

    with wp.ScopedDevice(args.device):
        with jax.default_device(wp.device_to_jax(wp.get_device())):
            example = Example(stage_path=args.stage_path, verbose=args.verbose)
            for _ in range(args.train_iters):
                example.step()
                example.render()
            if example.renderer:
                example.renderer.save()
            final_loss = float(example.forward())
            print(f"final_loss: {final_loss}")
