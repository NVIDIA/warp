import omni.kit.test
import omni.graph.core as og
from omni.graph.core.tests.omnigraph_test_utils import setup_test_environment
from omni.graph.core.tests.omnigraph_test_utils import verify_values
from omni.graph.core.tests.omnigraph_test_utils import load_test_file
import os
from contextlib import suppress


class TestOgnParticleSolver(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        await setup_test_environment()

    async def tearDown(self):
        await omni.usd.get_context().new_stage_async()

    async def test_ogn_omni_warp_OgnParticleSolver_import(self):
        import omni.warp.ogn.OgnParticleSolverDatabase

    async def test_ogn_TestNode_omni_warp_OgnParticleSolver_usda(self):
        test_file_name = "OgnParticleSolverTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        helper = og.OmniGraphHelper()
        test_node = helper.omnigraph_node("/Template_omni_warp_OgnParticleSolver")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 1)
        self.assertTrue(test_node.get_attribute_exists("inputs:collider"))

        self.assertTrue(test_node.get_attribute_exists("inputs:collider_offset"))

        input_attr = test_node.get_attribute("inputs:collider_offset")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(0.01, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:collider_offset attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:gravity"))

        input_attr = test_node.get_attribute("inputs:gravity")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values([0.0, -9.8, 0.0], actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:gravity attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:ground"))

        input_attr = test_node.get_attribute("inputs:ground")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(False, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:ground attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:ground_plane"))

        input_attr = test_node.get_attribute("inputs:ground_plane")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values([0.0, 1.0, 0.0], actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:ground_plane attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:k_contact_adhesion"))

        input_attr = test_node.get_attribute("inputs:k_contact_adhesion")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(0.75, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:k_contact_adhesion attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:k_contact_cohesion"))

        input_attr = test_node.get_attribute("inputs:k_contact_cohesion")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(0.75, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:k_contact_cohesion attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:k_contact_damp"))

        input_attr = test_node.get_attribute("inputs:k_contact_damp")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(100.0, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:k_contact_damp attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:k_contact_elastic"))

        input_attr = test_node.get_attribute("inputs:k_contact_elastic")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(5000.0, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:k_contact_elastic attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:k_contact_friction"))

        input_attr = test_node.get_attribute("inputs:k_contact_friction")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(2000.0, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:k_contact_friction attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:k_contact_mu"))

        input_attr = test_node.get_attribute("inputs:k_contact_mu")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(0.75, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:k_contact_mu attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:mass"))

        input_attr = test_node.get_attribute("inputs:mass")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(1.0, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:mass attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:num_substeps"))

        input_attr = test_node.get_attribute("inputs:num_substeps")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(32, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:num_substeps attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:positions"))

        input_attr = test_node.get_attribute("inputs:positions")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values([], actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:positions attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:radius"))

        input_attr = test_node.get_attribute("inputs:radius")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(10.0, actual_input, "omni.warp.OgnParticleSolver USD load test - inputs:radius attribute value error")
