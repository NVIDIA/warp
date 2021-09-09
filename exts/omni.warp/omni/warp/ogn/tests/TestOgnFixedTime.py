import omni.kit.test
import omni.graph.core as og
from omni.graph.core.tests.omnigraph_test_utils import setup_test_environment
from omni.graph.core.tests.omnigraph_test_utils import verify_values
from omni.graph.core.tests.omnigraph_test_utils import load_test_file
import os
from contextlib import suppress


class TestOgnFixedTime(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        await setup_test_environment()

    async def tearDown(self):
        await omni.usd.get_context().new_stage_async()

    async def test_ogn_omni_warp_FixedTime_import(self):
        import omni.warp.ogn.OgnFixedTimeDatabase

    async def test_ogn_TestNode_omni_warp_FixedTime_usda(self):
        test_file_name = "OgnFixedTimeTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        helper = og.OmniGraphHelper()
        test_node = helper.omnigraph_node("/Template_omni_warp_FixedTime")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_python_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 1)
        self.assertTrue(test_node.get_attribute_exists("inputs:end"))

        input_attr = test_node.get_attribute("inputs:end")
        actual_input = helper.get_values(test_node,  [input_attr])[0]
        verify_values(1000.0, actual_input, "omni.warp.FixedTime USD load test - inputs:end attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:fps"))

        input_attr = test_node.get_attribute("inputs:fps")
        actual_input = helper.get_values(test_node,  [input_attr])[0]
        verify_values(60.0, actual_input, "omni.warp.FixedTime USD load test - inputs:fps attribute value error")
        self.assertTrue(test_node.get_attribute_exists("inputs:start"))

        input_attr = test_node.get_attribute("inputs:start")
        actual_input = helper.get_values(test_node,  [input_attr])[0]
        verify_values(0.0, actual_input, "omni.warp.FixedTime USD load test - inputs:start attribute value error")
