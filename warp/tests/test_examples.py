import importlib
import os
import unittest

import warp as wp


# registers an example to run as a TestCase
def add_example_test(cls, name, options):
    def run(test, device):
        # disable scoped timer to avoid log spam
        wp.ScopedTimer.enabled = False

        try:
            module = importlib.import_module(f"examples.{name}")
        except Exception as e:
            print(f"Skipping example: {name}\n    Reason: {e}")
            return

        # create default USD stage output path which many examples expect
        if "stage" not in options:
            stage_path = os.path.join(os.path.dirname(__file__), f"outputs/{name}.usd")
            options["stage"] = stage_path

        if "num_frames" in options:
            num_frames = options["num_frames"]
            del options["num_frames"]
        else:
            num_frames = 10

        e = module.Example(**options)

        for _ in range(num_frames):
            e.update()
            e.render()

        wp.ScopedTimer.enabled = True

    from warp.tests.test_base import add_function_test

    add_function_test(cls, f"test_{name}", run, check_output=False)


def register(parent):
    class TestExamples(parent):
        pass

    # Exclude unless we can run headless somehow
    # add_example_test(TestExamples, name="example_render_opengl", options={})

    # TODO: Test CPU and GPU versions
    add_example_test(TestExamples, name="example_dem", options={})
    add_example_test(TestExamples, name="example_diffray", options={})
    add_example_test(TestExamples, name="example_fluid", options={})
    add_example_test(TestExamples, name="example_jacobian_ik", options={})
    add_example_test(TestExamples, name="example_marching_cubes", options={})
    add_example_test(TestExamples, name="example_mesh", options={})
    add_example_test(TestExamples, name="example_mesh_intersect", options={"num_frames": 1})
    add_example_test(TestExamples, name="example_nvdb", options={})
    add_example_test(TestExamples, name="example_raycast", options={})
    add_example_test(TestExamples, name="example_raymarch", options={})
    add_example_test(TestExamples, name="example_sim_cartpole", options={})
    add_example_test(TestExamples, name="example_sim_cloth", options={})
    add_example_test(TestExamples, name="example_sim_fk_grad", options={})
    add_example_test(TestExamples, name="example_sim_fk_grad_torch", options={})
    add_example_test(TestExamples, name="example_sim_grad_bounce", options={})
    add_example_test(TestExamples, name="example_sim_grad_cloth", options={})
    add_example_test(TestExamples, name="example_sim_granular", options={})
    add_example_test(TestExamples, name="example_sim_granular_collision_sdf", options={})
    add_example_test(TestExamples, name="example_sim_neo_hookean", options={})
    add_example_test(TestExamples, name="example_sim_particle_chain", options={})
    add_example_test(TestExamples, name="example_sim_quadruped", options={})
    add_example_test(TestExamples, name="example_sim_rigid_chain", options={})
    add_example_test(TestExamples, name="example_sim_rigid_contact", options={})
    add_example_test(TestExamples, name="example_sim_rigid_fem", options={})
    add_example_test(TestExamples, name="example_sim_rigid_force", options={})
    add_example_test(TestExamples, name="example_sim_rigid_gyroscopic", options={})
    add_example_test(TestExamples, name="example_sim_rigid_kinematics", options={})
    add_example_test(TestExamples, name="example_sim_trajopt", options={})
    add_example_test(TestExamples, name="example_sph", options={})
    add_example_test(TestExamples, name="example_wave", options={"resx": 256, "resy": 256})

    return TestExamples


if __name__ == "__main__":
    wp.init()

    # force rebuild of all kernels
    wp.build.clear_kernel_cache()

    c = register(unittest.TestCase)

    unittest.main(verbosity=2, failfast=True)
