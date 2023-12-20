# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import importlib
import os
import unittest

import warp as wp
from warp.tests.unittest_utils import get_unique_cuda_test_devices, sanitize_identifier

wp.init()


# registers an example to run as a TestCase
def add_example_test(cls, name, devices=None, options={}):
    def run(test, device):
        # The copy() is needed because pop() is used to avoid passing extra args to Example()
        # Can remove once all examples accept **kwargs and we no longer need to pop()
        test_options = options.copy()

        try:
            module = importlib.import_module(f"examples.{name}")

            torch_cuda_required = test_options.setdefault("torch_cuda_required", False)
            test_options.pop("torch_cuda_required", None)
            if torch_cuda_required and wp.get_device(device).is_cuda:
                # Ensure torch has CUDA support
                import torch

                if not torch.cuda.is_available():
                    test.skipTest("Torch not compiled with CUDA support")

        except Exception as e:
            test.skipTest(f"{e}")

        # create default USD stage output path which many examples expect
        test_options.setdefault(
            "stage", os.path.join(os.path.dirname(__file__), f"outputs/{name}_{sanitize_identifier(device)}.usd")
        )

        try:
            os.remove(test_options["stage"])
        except OSError:
            pass

        num_frames = test_options.get("num_frames", 10)
        test_options.pop("num_frames", None)

        # Don't want to force load all modules by default for serial test runner
        wp.config.graph_capture_module_load_default = False

        try:
            enable_backward = test_options.get("enable_backward", True)
            wp.set_module_options({"enable_backward": enable_backward}, module)
            test_options.pop("enable_backward", None)

            with wp.ScopedDevice(device):
                wp.load_module(module, device=wp.get_device())
                extra_load_modules = test_options.get("load_modules", [])
                for module_name in extra_load_modules:
                    wp.load_module(module_name, device=wp.get_device())
                test_options.pop("load_modules", None)

                e = module.Example(**test_options)

                # disable scoped timer to avoid log spam from time steps
                wp.ScopedTimer.enabled = False

                for _ in range(num_frames):
                    e.update()
                    e.render()
        except Exception as e:
            test.fail(f"{e}")
        finally:
            wp.ScopedTimer.enabled = True
            wp.config.graph_capture_module_load_default = True

    from warp.tests.unittest_utils import add_function_test

    add_function_test(cls, f"test_{name}", run, devices=devices, check_output=False)


# TODO: Make the example classes use the passed in device
cuda_test_devices = get_unique_cuda_test_devices()

# NOTE: To give the parallel test runner more opportunities to parallelize test cases,
# we break up the tests into multiple TestCase classes that should be non-conflicting
# w.r.t. kernel compilation


class TestExamples(unittest.TestCase):
    pass


# Exclude unless we can run headless somehow
# add_example_test(TestExamples, name="example_render_opengl", options={})

add_example_test(TestExamples, name="example_dem", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_diffray", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_fluid", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_jacobian_ik", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_marching_cubes", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_mesh", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_mesh_intersect", devices=cuda_test_devices, options={"num_frames": 1})
add_example_test(TestExamples, name="example_nvdb", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_raycast", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_raymarch", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_sph", devices=cuda_test_devices)
add_example_test(TestExamples, name="example_wave", devices=cuda_test_devices, options={"resx": 256, "resy": 256})


class TestSimExamples(unittest.TestCase):
    pass


add_example_test(
    TestSimExamples,
    name="example_sim_cartpole",
    devices=cuda_test_devices,
    options={"load_modules": ["warp.sim.collide", "warp.sim.integrator_euler", "warp.sim.articulation"]},
)
add_example_test(
    TestSimExamples,
    name="example_sim_cloth",
    devices=cuda_test_devices,
    options={"load_modules": ["warp.sim.collide", "warp.sim.integrator_euler", "warp.sim.particles"]},
)
add_example_test(TestSimExamples, name="example_sim_fk_grad", devices=cuda_test_devices)
add_example_test(
    TestSimExamples, name="example_sim_fk_grad_torch", devices=cuda_test_devices, options={"torch_cuda_required": True}
)
add_example_test(
    TestSimExamples,
    name="example_sim_grad_bounce",
    devices=cuda_test_devices,
    options={"load_modules": ["warp.sim.integrator_euler", "warp.sim.particles"]},
)
add_example_test(
    TestSimExamples,
    name="example_sim_grad_cloth",
    devices=cuda_test_devices,
    options={"load_modules": ["warp.sim.integrator_euler", "warp.sim.particles"]},
)
add_example_test(TestSimExamples, name="example_sim_granular", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="example_sim_granular_collision_sdf", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="example_sim_neo_hookean", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="example_sim_particle_chain", devices=cuda_test_devices)
add_example_test(
    TestSimExamples,
    name="example_sim_quadruped",
    devices=cuda_test_devices,
    options={"load_modules": ["warp.sim.integrator_xpbd", "warp.sim.integrator_euler"]},
)
add_example_test(
    TestSimExamples,
    name="example_sim_rigid_chain",
    devices=cuda_test_devices,
    options={"load_modules": ["warp.sim.integrator_xpbd", "warp.sim.integrator_euler"]},
)
add_example_test(
    TestSimExamples,
    name="example_sim_rigid_contact",
    devices=cuda_test_devices,
    options={"load_modules": ["warp.sim.integrator_euler"]},
)
add_example_test(TestSimExamples, name="example_sim_rigid_fem", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="example_sim_rigid_force", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="example_sim_rigid_gyroscopic", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="example_sim_rigid_kinematics", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="example_sim_trajopt", devices=cuda_test_devices)


class TestFemExamples(unittest.TestCase):
    pass


add_example_test(
    TestFemExamples,
    name="fem.example_diffusion_mgpu",
    devices=cuda_test_devices,
    options={"quiet": True, "num_frames": 1, "enable_backward": False},
)

# The following examples do not need CUDA, but they need USD
add_example_test(
    TestFemExamples,
    name="fem.example_apic_fluid",
    devices=cuda_test_devices,
    options={"quiet": True, "res": [16, 16, 16], "enable_backward": False},
)
add_example_test(
    TestFemExamples,
    name="fem.example_diffusion",
    devices=cuda_test_devices,
    options={"quiet": True, "resolution": 10, "mesh": "tri", "num_frames": 1, "enable_backward": False},
)
add_example_test(
    TestFemExamples,
    name="fem.example_diffusion_3d",
    devices=cuda_test_devices,
    options={"quiet": True, "resolution": 10, "num_frames": 1, "enable_backward": False},
)
add_example_test(
    TestFemExamples,
    name="fem.example_deformed_geometry",
    devices=cuda_test_devices,
    options={"quiet": True, "resolution": 10, "num_frames": 1, "mesh": "tri", "enable_backward": False},
)
add_example_test(
    TestFemExamples,
    name="fem.example_convection_diffusion",
    devices=cuda_test_devices,
    options={"quiet": True, "resolution": 20, "enable_backward": False},
)
add_example_test(
    TestFemExamples,
    name="fem.example_mixed_elasticity",
    devices=cuda_test_devices,
    options={"quiet": True, "nonconforming_stresses": True, "mesh": "quad", "num_frames": 1, "enable_backward": False},
)
add_example_test(
    TestFemExamples,
    name="fem.example_stokes_transfer",
    devices=cuda_test_devices,
    options={"quiet": True, "num_frames": 1, "enable_backward": False},
)


if __name__ == "__main__":
    # force rebuild of all kernels
    wp.build.clear_kernel_cache()

    unittest.main(verbosity=2, failfast=True)
