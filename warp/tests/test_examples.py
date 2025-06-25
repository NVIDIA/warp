# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test Warp examples with unittest.

This module tests the Warp examples registered in it using the unittest
framework. When registering tests with add_example_test(), three optional
dictionaries can be provided: test_options, test_options_cuda, and
test_options_cpu. These are added to the command line arguments in-order, so
if a parameter is specified in both test_options and test_options_cuda, the
one in test_options_cuda will take precedence due to how argparse works.

Generally the test_options[_cpu,_cuda] dictionaries should be used to prevent
graphical windows from being open by the example {"headless": True} and to
override example defaults so the example can run in less than ten seconds.

To skip tests if the optional dependencies are not found, use the following keys:
- {"usd_required": True} (requires usd-core)
- {"torch_required": True} (requires torch)
- {"pillow_required": True} (requires pillow)

Use the "num_frames" and "train_iters" keys to control the number of steps.

Use "test_timeout" to override the default test timeout threshold of 600 seconds.
"""

import os
import platform
import subprocess
import sys
import unittest
from typing import Any, Dict, Optional, Type

import warp as wp
import warp.tests.unittest_utils
from warp.tests.unittest_utils import (
    USD_AVAILABLE,
    get_selected_cuda_test_devices,
    get_test_devices,
    sanitize_identifier,
)
from warp.utils import check_p2p

wp.init()  # For wp.context.runtime.core.wp_is_debug_enabled()


def _build_command_line_options(test_options: Dict[str, Any]) -> list:
    """Helper function to build command-line options from the test options dictionary."""
    additional_options = []

    for key, value in test_options.items():
        if key == "headless" and value:
            additional_options.extend(["--headless"])
        else:
            # Just add --key value
            additional_options.extend(["--" + key, str(value)])

    return additional_options


def _merge_options(base_options: Dict[str, Any], device_options: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to merge base test options with device-specific test options."""
    merged_options = base_options.copy()

    #  Update options with device-specific dictionary, overwriting existing keys with the more-specific values
    merged_options.update(device_options)
    return merged_options


def add_example_test(
    cls: Type,
    name: str,
    devices: Optional[list] = None,
    test_options: Optional[Dict[str, Any]] = None,
    test_options_cpu: Optional[Dict[str, Any]] = None,
    test_options_cuda: Optional[Dict[str, Any]] = None,
):
    """Registers a Warp example to run on ``devices`` as a TestCase."""

    if test_options is None:
        test_options = {}
    if test_options_cpu is None:
        test_options_cpu = {}
    if test_options_cuda is None:
        test_options_cuda = {}

    def run(test, device):
        if wp.get_device(device).is_cuda:
            options = _merge_options(test_options, test_options_cuda)
        else:
            options = _merge_options(test_options, test_options_cpu)

        # Mark the test as skipped if Torch is not installed but required
        torch_required = options.pop("torch_required", False)
        if torch_required:
            try:
                import torch

                if wp.get_device(device).is_cuda and not torch.cuda.is_available():
                    # Ensure torch has CUDA support
                    test.skipTest("Torch not compiled with CUDA support")

            except Exception as e:
                test.skipTest(f"{e}")

        # Mark the test as skipped if USD is not installed but required
        usd_required = options.pop("usd_required", False)
        if usd_required and not USD_AVAILABLE:
            test.skipTest("Requires usd-core")

        # Mark the test as skipped if pillow is not installed but required
        pillow_required = options.pop("pillow_required", False)
        if pillow_required:
            try:
                import PIL  # noqa: F401
            except ImportError:
                test.skipTest("Requires pillow")

        # Find the current Warp cache
        warp_cache_path = wp.config.kernel_cache_dir

        env_vars = os.environ.copy()
        if warp_cache_path is not None:
            env_vars["WARP_CACHE_PATH"] = warp_cache_path

        if warp.tests.unittest_utils.coverage_enabled:
            import tempfile

            # Generate a random coverage data file name - file is deleted along with containing directory
            with tempfile.NamedTemporaryFile(
                dir=warp.tests.unittest_utils.coverage_temp_dir, delete=False
            ) as coverage_file:
                pass

            command = ["coverage", "run", f"--data-file={coverage_file.name}"]

            if warp.tests.unittest_utils.coverage_branch:
                command.append("--branch")

        else:
            command = [sys.executable]

        # Append Warp commands
        command.extend(["-m", f"warp.examples.{name}", "--device", str(device)])

        stage_path = (
            options.pop(
                "stage_path",
                os.path.join(os.path.dirname(__file__), f"outputs/{name}_{sanitize_identifier(device)}.usd"),
            )
            if USD_AVAILABLE
            else "None"
        )

        if stage_path:
            command.extend(["--stage_path", stage_path])
            try:
                os.remove(stage_path)
            except OSError:
                pass

        command.extend(_build_command_line_options(options))

        # Set the test timeout in seconds
        test_timeout = options.pop("test_timeout", 600)

        # with wp.ScopedTimer(f"{name}_{sanitize_identifier(device)}"):
        # Run the script as a subprocess
        result = subprocess.run(
            command, capture_output=True, text=True, env=env_vars, timeout=test_timeout, check=False
        )

        # Check the return code (0 is standard for success)
        test.assertEqual(
            result.returncode,
            0,
            msg=f"Failed with return code {result.returncode}, command: {' '.join(command)}\n\nOutput:\n{result.stdout}\n{result.stderr}",
        )

        # If the test succeeded, try to clean up the output by default
        if stage_path and result.returncode == 0:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    from warp.tests.unittest_utils import add_function_test

    add_function_test(cls, f"test_{name}", run, devices=devices, check_output=False)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")

# NOTE: To give the parallel test runner more opportunities to parallelize test cases,
# we break up the tests into multiple TestCase classes


class TestCoreExamples(unittest.TestCase):
    pass


# Exclude unless we can run headless somehow
# add_example_test(TestCoreExamples, name="example_render_opengl")

add_example_test(TestCoreExamples, name="core.example_dem", devices=test_devices, test_options_cpu={"num_frames": 2})
add_example_test(
    TestCoreExamples,
    name="core.example_fluid",
    devices=test_devices,
    test_options={"num_frames": 100, "headless": True},
)
add_example_test(
    TestCoreExamples,
    name="core.example_graph_capture",
    devices=test_devices,
    test_options={"headless": True},
    test_options_cpu={"num_frames": 100},
)
add_example_test(TestCoreExamples, name="core.example_marching_cubes", devices=cuda_test_devices)
add_example_test(TestCoreExamples, name="core.example_mesh", devices=test_devices, test_options={"usd_required": True})
add_example_test(
    TestCoreExamples, name="core.example_mesh_intersect", devices=test_devices, test_options={"usd_required": True}
)
add_example_test(TestCoreExamples, name="core.example_nvdb", devices=test_devices)
add_example_test(
    TestCoreExamples,
    name="core.example_raycast",
    devices=test_devices,
    test_options={"usd_required": True, "headless": True},
)
if platform.system() == "Windows":
    # Skip GPU testing because of obscure NVRTC bug with illegal memory access
    add_example_test(
        TestCoreExamples,
        name="core.example_raymarch",
        devices=[wp.get_device("cpu")],
        test_options={"height": 512, "width": 1024, "headless": True},
    )
else:
    add_example_test(
        TestCoreExamples,
        name="core.example_raymarch",
        devices=test_devices,
        test_options={"height": 512, "width": 1024, "headless": True},
    )
add_example_test(
    TestCoreExamples,
    name="core.example_sample_mesh",
    devices=test_devices,
    test_options_cpu={"num_frames": 1},
)
add_example_test(
    TestCoreExamples,
    name="core.example_sph",
    devices=test_devices,
    test_options_cpu={"num_frames": 1},
)
add_example_test(
    TestCoreExamples,
    name="core.example_torch",
    devices=test_devices,
    test_options={"headless": True, "num_frames": 1000, "torch_required": True},
)
add_example_test(TestCoreExamples, name="core.example_wave", devices=test_devices)


class TestOptimExamples(unittest.TestCase):
    pass


add_example_test(
    TestOptimExamples,
    name="optim.example_bounce",
    devices=test_devices,
    test_options_cpu={"train_iters": 3},
)
add_example_test(
    TestOptimExamples,
    name="optim.example_drone",
    devices=test_devices,
    test_options={"headless": True},
    test_options_cpu={"num_frames": 10},
)
add_example_test(
    TestOptimExamples,
    name="optim.example_cloth_throw",
    devices=test_devices,
    test_options_cpu={"train_iters": 3},
)
add_example_test(
    TestOptimExamples,
    name="optim.example_diffray",
    devices=test_devices,
    test_options={"usd_required": True, "headless": True},
    test_options_cpu={"train_iters": 2},
)
add_example_test(
    TestOptimExamples,
    name="optim.example_fluid_checkpoint",
    devices=cuda_test_devices,
    test_options={"headless": True, "train_iters": 5, "num_frames": 300, "pillow_required": True},
)
add_example_test(TestOptimExamples, name="optim.example_inverse_kinematics", devices=test_devices)
add_example_test(
    TestOptimExamples,
    name="optim.example_inverse_kinematics_torch",
    devices=test_devices,
    test_options={"torch_required": True},
)
add_example_test(TestOptimExamples, name="optim.example_spring_cage", devices=test_devices)
add_example_test(
    TestOptimExamples,
    name="optim.example_trajectory",
    devices=test_devices,
    test_options={"headless": True, "train_iters": 50},
)
add_example_test(
    TestOptimExamples,
    name="optim.example_softbody_properties",
    devices=test_devices,
    test_options_cuda={
        "train_iters": 1 if warp.context.runtime.core.wp_is_debug_enabled() else 3,
    },
    test_options_cpu={"train_iters": 1},
)


class TestSimExamples(unittest.TestCase):
    pass


add_example_test(TestSimExamples, name="sim.example_cartpole", devices=test_devices)
add_example_test(
    TestSimExamples,
    name="sim.example_cloth",
    devices=test_devices,
    test_options={"usd_required": True},
    test_options_cpu={"num_frames": 10},
)
add_example_test(
    TestSimExamples, name="sim.example_granular", devices=test_devices, test_options_cpu={"num_frames": 10}
)
add_example_test(TestSimExamples, name="sim.example_granular_collision_sdf", devices=cuda_test_devices)
add_example_test(TestSimExamples, name="sim.example_jacobian_ik", devices=test_devices)
add_example_test(TestSimExamples, name="sim.example_particle_chain", devices=test_devices)
add_example_test(
    TestSimExamples, name="sim.example_quadruped", devices=test_devices, test_options_cpu={"num_frames": 100}
)
add_example_test(TestSimExamples, name="sim.example_rigid_chain", devices=test_devices)
add_example_test(
    TestSimExamples,
    name="sim.example_rigid_contact",
    devices=test_devices,
    test_options={"usd_required": True},
    test_options_cpu={"num_frames": 3},
)
add_example_test(
    TestSimExamples, name="sim.example_rigid_soft_contact", devices=test_devices, test_options_cpu={"num_frames": 10}
)
add_example_test(TestSimExamples, name="sim.example_rigid_force", devices=test_devices)
add_example_test(TestSimExamples, name="sim.example_rigid_gyroscopic", devices=test_devices)
add_example_test(
    TestSimExamples, name="sim.example_soft_body", devices=test_devices, test_options_cpu={"num_frames": 10}
)


class TestFemExamples(unittest.TestCase):
    pass


class TestFemDiffusionExamples(unittest.TestCase):
    pass


# MGPU tests may fail on systems where P2P transfers are misconfigured
if check_p2p():
    add_example_test(
        TestFemDiffusionExamples,
        name="fem.example_diffusion_mgpu",
        devices=get_selected_cuda_test_devices(mode="basic"),
        test_options={"headless": True},
    )

add_example_test(
    TestFemExamples,
    name="fem.example_apic_fluid",
    devices=get_selected_cuda_test_devices(mode="basic"),
    test_options={"num_frames": 5, "voxel_size": 2.0},
)
add_example_test(
    TestFemExamples,
    name="fem.example_adaptive_grid",
    devices=get_selected_cuda_test_devices(mode="basic"),
    test_options={"headless": True, "div_conforming": True},
)

# The following examples do not need CUDA
add_example_test(
    TestFemDiffusionExamples,
    name="fem.example_diffusion",
    devices=test_devices,
    test_options={"resolution": 10, "mesh": "tri", "headless": True},
)
add_example_test(
    TestFemDiffusionExamples, name="fem.example_diffusion_3d", devices=test_devices, test_options={"headless": True}
)
add_example_test(
    TestFemExamples,
    name="fem.example_deformed_geometry",
    devices=test_devices,
    test_options={"resolution": 10, "mesh": "tri", "headless": True},
)
add_example_test(
    TestFemExamples,
    name="fem.example_convection_diffusion",
    devices=test_devices,
    test_options={"resolution": 20, "headless": True},
)
add_example_test(
    TestFemExamples,
    name="fem.example_burgers",
    devices=test_devices,
    test_options={"resolution": 20, "num_frames": 25, "degree": 1, "headless": True},
)
add_example_test(
    TestFemExamples,
    name="fem.example_convection_diffusion_dg",
    devices=test_devices,
    test_options={"resolution": 20, "num_frames": 25, "headless": True},
)
add_example_test(
    TestFemExamples,
    name="fem.example_mixed_elasticity",
    devices=test_devices,
    test_options={"nonconforming_stresses": True, "mesh": "quad", "headless": True},
)
add_example_test(
    TestFemExamples, name="fem.example_stokes_transfer", devices=test_devices, test_options={"headless": True}
)
add_example_test(
    TestFemExamples,
    name="fem.example_stokes",
    devices=test_devices,
    test_options={"resolution": 10, "nonconforming_pressures": True, "headless": True},
)
add_example_test(
    TestFemExamples,
    name="fem.example_navier_stokes",
    devices=test_devices,
    test_options={"num_frames": 101, "resolution": 10, "tri_mesh": True, "headless": True},
)
add_example_test(
    TestFemExamples,
    name="fem.example_streamlines",
    devices=get_selected_cuda_test_devices(),
    test_options={"headless": True},
)
add_example_test(
    TestFemExamples,
    name="fem.example_distortion_energy",
    devices=get_selected_cuda_test_devices(),
    test_options={"headless": True, "resolution": 16},
)
add_example_test(
    TestFemExamples,
    name="fem.example_magnetostatics",
    devices=test_devices,
    test_options={"headless": True, "resolution": 16},
)
add_example_test(
    TestFemExamples,
    name="fem.example_nonconforming_contact",
    devices=test_devices,
    test_options={"headless": True, "resolution": 16, "num_steps": 2},
)

if __name__ == "__main__":
    # force rebuild of all kernels
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
