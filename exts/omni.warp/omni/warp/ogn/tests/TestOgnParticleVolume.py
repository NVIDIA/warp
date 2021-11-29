import omni.kit.test
import omni.graph.core as og
from omni.graph.core.tests.omnigraph_test_utils import setup_test_environment
from omni.graph.core.tests.omnigraph_test_utils import verify_values
from omni.graph.core.tests.omnigraph_test_utils import load_test_file
import os
from contextlib import suppress


class TestOgnParticleVolume(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        await setup_test_environment()

    async def tearDown(self):
        await omni.usd.get_context().new_stage_async()

    async def test_ogn_omni_warp_OgnParticleVolume_import(self):
        import omni.warp.ogn.OgnParticleVolumeDatabase

    async def test_ogn_TestNode_omni_warp_OgnParticleVolume_usda(self):
        test_file_name = "OgnParticleVolumeTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        helper = og.OmniGraphHelper()
        test_node = helper.omnigraph_node("/Template_omni_warp_OgnParticleVolume")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 1)
        self.assertTrue(test_node.get_attribute_exists("inputs:execIn"))

        input_attr = test_node.get_attribute("inputs:execIn")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(0, actual_input, "omni.warp.OgnParticleVolume USD load test - inputs:execIn attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:sample_surface"))

        input_attr = test_node.get_attribute("inputs:sample_surface")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(True, actual_input, "omni.warp.OgnParticleVolume USD load test - inputs:sample_surface attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:sample_volume"))

        input_attr = test_node.get_attribute("inputs:sample_volume")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(True, actual_input, "omni.warp.OgnParticleVolume USD load test - inputs:sample_volume attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:spacing"))

        input_attr = test_node.get_attribute("inputs:spacing")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(10.0, actual_input, "omni.warp.OgnParticleVolume USD load test - inputs:spacing attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:spacing_jitter"))

        input_attr = test_node.get_attribute("inputs:spacing_jitter")
        actual_input = helper.get_values(test_node, [input_attr])[0]
        verify_values(0.0, actual_input, "omni.warp.OgnParticleVolume USD load test - inputs:spacing_jitter attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:volume"))

