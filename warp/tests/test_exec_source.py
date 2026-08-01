# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import linecache
import subprocess
import sys
import types
import unittest

import numpy as np

import warp as wp
from warp._src import context
from warp._src.codegen import WarpCodegenKeyError

failed_payload_ref = None


@wp.kernel
def file_defined_increment(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] + 1.0


class TestExecSource(unittest.TestCase):
    """Test execution of trusted source containing Warp definitions."""

    def test_public_export(self):
        self.assertIs(wp.exec_source, context.exec_source)

    def test_kernel_with_array_and_scalar_arguments(self):
        generated = wp.exec_source(
            """
@wp.kernel
def scale(values: wp.array(dtype=wp.float32), factor: float):
    i = wp.tid()
    values[i] = values[i] * factor
""",
            module_name="warp.tests.exec_source.array_scalar",
        )
        values = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        wp.launch(generated["scale"], dim=values.shape, inputs=[values, 3.0], device="cpu")
        np.testing.assert_allclose(values.numpy(), np.array([3.0, 6.0, 9.0], dtype=np.float32))

    def test_subprocess_workflow(self):
        code = """
import numpy as np
import warp as wp
from warp._src import context

generated = wp.exec_source('''
@wp.kernel
def double(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] * 2.0
''', module_name='warp.tests.exec_source.subprocess')
values = wp.array([1.0, 2.0], dtype=wp.float32, device='cpu')
wp.launch(generated['double'], dim=2, inputs=[values], device='cpu')
np.testing.assert_allclose(values.numpy(), np.array([2.0, 4.0], dtype=np.float32))
print('OK')
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_multiple_kernels(self):
        generated = wp.exec_source(
            """
@wp.kernel
def add_one(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] + 1.0

@wp.kernel
def multiply_two(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] * 2.0
""",
            module_name="warp.tests.exec_source.multiple_kernels",
        )
        values = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        wp.launch(generated["add_one"], dim=values.shape, inputs=[values], device="cpu")
        wp.launch(generated["multiply_two"], dim=values.shape, inputs=[values], device="cpu")
        np.testing.assert_allclose(values.numpy(), np.array([4.0, 6.0], dtype=np.float32))

    def test_function_struct_import_and_annotations(self):
        generated = wp.exec_source(
            """
import math

@wp.struct
class Parameters:
    factor: wp.float32

@wp.func
def apply_factor(value: wp.float32, parameters: Parameters):
    return value * parameters.factor

@wp.kernel
def transform(values: wp.array(dtype=wp.float32), parameters: Parameters):
    i = wp.tid()
    values[i] = apply_factor(values[i], parameters)

rounded = math.ceil(1.25)
""",
            module_name="warp.tests.exec_source.constructs",
        )
        parameters = generated["Parameters"]()
        parameters.factor = 4.0
        values = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        wp.launch(generated["transform"], dim=values.shape, inputs=[values, parameters], device="cpu")
        np.testing.assert_allclose(values.numpy(), np.array([4.0, 8.0], dtype=np.float32))
        self.assertEqual(generated["rounded"], 2)
        self.assertIn("math", generated)

    def test_input_validation(self):
        with self.assertRaisesRegex(TypeError, "source must be a string"):
            wp.exec_source(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "module name must be a string or None"):
            wp.exec_source("", module_name=1)  # type: ignore[arg-type]
        for module_name in ("", "invalid-name", "valid.invalid-name", "valid.class"):
            with self.subTest(module_name=module_name):
                with self.assertRaisesRegex(ValueError, "valid dotted Python identifier"):
                    wp.exec_source("", module_name=module_name)

    def test_syntax_error_location(self):
        module_name = "warp.tests.exec_source.syntax_error"
        source = """value = 1
@wp.kernel
def broken(values: wp.array(dtype=wp.float32)):
    values[0] =
"""
        with self.assertRaises(SyntaxError) as raised:
            wp.exec_source(source, module_name=module_name)
        self.assertTrue(raised.exception.filename.startswith(f"<warp-source:{module_name}:"))
        self.assertEqual(raised.exception.lineno, 4)
        self.assertNotIn(module_name, context.user_modules)
        self.assertNotIn(module_name, context._generated_source_modules)

    def test_codegen_error_location(self):
        module_name = "warp.tests.exec_source.codegen_error"
        generated = wp.exec_source(
            """
@wp.kernel
def broken(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = missing_function(i)
""",
            module_name=module_name,
        )
        values = wp.zeros(1, dtype=wp.float32, device="cpu")
        with self.assertRaises(WarpCodegenKeyError) as raised:
            wp.launch(generated["broken"], dim=1, inputs=[values], device="cpu")
        message = str(raised.exception)
        self.assertIn(f"<warp-source:{module_name}:", message)
        self.assertIn(":5", message)
        self.assertIn("missing_function", message)

    def test_runtime_error_before_registration(self):
        module_name = "warp.tests.exec_source.early_failure"
        with self.assertRaisesRegex(RuntimeError, "early failure"):
            wp.exec_source('raise RuntimeError("early failure")', module_name=module_name)
        self.assertNotIn(module_name, context.user_modules)
        self.assertNotIn(module_name, context._generated_source_modules)

    def test_system_exit_after_registration_is_atomic(self):
        module_name = "warp.tests.exec_source.system_exit"
        source = """
@wp.kernel
def partial(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i]

raise SystemExit(7)
"""
        with self.assertRaises(SystemExit):
            wp.exec_source(source, module_name=module_name)
        self.assertNotIn(module_name, context.user_modules)
        self.assertNotIn(module_name, context._generated_source_modules)

    def test_failure_after_registration_is_atomic(self):
        module_name = "warp.tests.exec_source.partial_failure"
        source = """
@wp.kernel
def partial(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i]

raise RuntimeError("failure after registration")
"""
        try:
            wp.exec_source(source, module_name=module_name)
        except RuntimeError as error:
            self.assertIn("failure after registration", str(error))
            traceback = error.__traceback__
        else:
            self.fail("Expected source execution to fail")
        self.assertIsNotNone(traceback)
        while traceback.tb_next is not None:
            traceback = traceback.tb_next
        self.assertTrue(traceback.tb_frame.f_code.co_filename.startswith(f"<warp-source:{module_name}:"))
        self.assertNotIn(module_name, context.user_modules)
        self.assertNotIn(module_name, context._generated_source_modules)
        self.assertFalse(any(name.startswith(f"<warp-source:{module_name}:") for name in linecache.cache))

    def test_failed_source_does_not_retain_its_namespace(self):
        global failed_payload_ref

        module_name = "warp.tests.exec_source.failed_namespace"
        source = f"""
import weakref
import importlib

test_module = importlib.import_module({__name__!r})

class Payload:
    pass

payload = Payload()
test_module.failed_payload_ref = weakref.ref(payload)

@wp.kernel
def partial(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i]

raise RuntimeError("discard namespace")
"""
        with self.assertRaisesRegex(RuntimeError, "discard namespace"):
            wp.exec_source(source, module_name=module_name)

        gc.collect()
        self.assertIsNotNone(failed_payload_ref)
        self.assertIsNone(failed_payload_ref())
        failed_payload_ref = None

    def test_collision_with_file_backed_module(self):
        with self.assertRaisesRegex(ValueError, "collides with an existing module"):
            wp.exec_source("value = 1", module_name=__name__)
        self.assertIs(wp.get_module(__name__), file_defined_increment.module)

    def test_default_name_reuse(self):
        source = """
@wp.kernel
def default_identity(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i]
"""
        count_before = len(context._generated_source_modules)
        first = wp.exec_source(source)
        second = wp.exec_source(source)
        self.assertIs(first, second)
        self.assertTrue(first["default_identity"].module.name.startswith("__warp_source_"))
        self.assertEqual(len(context._generated_source_modules), count_before + 1)

    def test_success_removes_linecache_entry(self):
        module_name = "warp.tests.exec_source.linecache_success"
        generated = wp.exec_source(
            """
@wp.kernel
def identity(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i]
""",
            module_name=module_name,
        )
        synthetic_filename = generated["identity"].adj.filename
        self.assertTrue(synthetic_filename.startswith(f"<warp-source:{module_name}:"))
        self.assertNotIn(synthetic_filename, linecache.cache)

    def test_result_is_read_only_and_excludes_reserved_names(self):
        generated = wp.exec_source("value = 42", module_name="warp.tests.exec_source.result_mapping")
        self.assertIsInstance(generated, types.MappingProxyType)
        self.assertEqual(generated["value"], 42)
        for reserved_name in ("__builtins__", "__file__", "__name__", "__package__", "wp"):
            self.assertNotIn(reserved_name, generated)
        with self.assertRaises(TypeError):
            generated["value"] = 0  # type: ignore[index]

    def test_explicit_name_reuse(self):
        source = "value = object()"
        module_name = "warp.tests.exec_source.explicit_reuse"
        first = wp.exec_source(source, module_name=module_name)
        second = wp.exec_source(source, module_name=module_name)
        self.assertIs(first, second)
        self.assertIs(first["value"], second["value"])

    def test_changed_source_is_rejected_without_mutation(self):
        module_name = "warp.tests.exec_source.changed_rejected"
        first = wp.exec_source("value = 1", module_name=module_name)
        with self.assertRaisesRegex(ValueError, "different source"):
            wp.exec_source("value = 2", module_name=module_name)
        self.assertIs(context._generated_source_modules[module_name].result, first)
        self.assertEqual(first["value"], 1)

    def test_same_symbol_in_different_modules(self):
        first = wp.exec_source(
            """
@wp.kernel
def shared(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] * 2.0
""",
            module_name="warp.tests.exec_source.shared_a",
        )
        second = wp.exec_source(
            """
@wp.kernel
def shared(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] * 3.0
""",
            module_name="warp.tests.exec_source.shared_b",
        )
        first_values = wp.array([2.0], dtype=wp.float32, device="cpu")
        second_values = wp.array([2.0], dtype=wp.float32, device="cpu")
        wp.launch(first["shared"], dim=1, inputs=[first_values], device="cpu")
        wp.launch(second["shared"], dim=1, inputs=[second_values], device="cpu")
        np.testing.assert_allclose(first_values.numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_allclose(second_values.numpy(), np.array([6.0], dtype=np.float32))
        self.assertIsNot(first["shared"].module, second["shared"].module)

    def test_changed_body_does_not_launch_stale_executable(self):
        template = """
@wp.kernel
def operation(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i] * {factor}.0
"""
        first = wp.exec_source(template.format(factor=2), module_name="warp.tests.exec_source.body_a")
        second = wp.exec_source(template.format(factor=5), module_name="warp.tests.exec_source.body_b")
        first_values = wp.array([3.0], dtype=wp.float32, device="cpu")
        second_values = wp.array([3.0], dtype=wp.float32, device="cpu")
        wp.launch(first["operation"], dim=1, inputs=[first_values], device="cpu")
        wp.launch(second["operation"], dim=1, inputs=[second_values], device="cpu")
        np.testing.assert_allclose(first_values.numpy(), np.array([6.0], dtype=np.float32))
        np.testing.assert_allclose(second_values.numpy(), np.array([15.0], dtype=np.float32))

    def test_dependency_changes_affect_module_hash(self):
        template = """
@wp.struct
class Parameters:
    factor: wp.float32
    {extra_field}

@wp.func
def helper(value: wp.float32, parameters: Parameters):
    return value * parameters.factor {helper_suffix}

@wp.kernel
def operation(values: wp.array(dtype=wp.float32), parameters: Parameters):
    i = wp.tid()
    values[i] = helper(values[i], parameters)
"""
        unchanged_a = wp.exec_source(
            template.format(extra_field="", helper_suffix=""), module_name="warp.tests.exec_source.hash_base_a"
        )
        unchanged_b = wp.exec_source(
            template.format(extra_field="", helper_suffix=""), module_name="warp.tests.exec_source.hash_base_b"
        )
        helper_changed = wp.exec_source(
            template.format(extra_field="", helper_suffix="+ 1.0"),
            module_name="warp.tests.exec_source.hash_helper",
        )
        struct_changed = wp.exec_source(
            template.format(extra_field="offset: wp.float32", helper_suffix=""),
            module_name="warp.tests.exec_source.hash_struct",
        )
        base_hash = unchanged_a["operation"].module.hash_module()
        self.assertEqual(base_hash, unchanged_b["operation"].module.hash_module())
        self.assertNotEqual(base_hash, helper_changed["operation"].module.hash_module())
        self.assertNotEqual(base_hash, struct_changed["operation"].module.hash_module())

    def test_file_defined_kernel_and_lookup_are_unaffected(self):
        original_module = wp.get_module(__name__)
        wp.exec_source(
            """
@wp.kernel
def generated_identity(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = values[i]
""",
            module_name="warp.tests.exec_source.file_kernel_control",
        )
        values = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        wp.launch(file_defined_increment, dim=values.shape, inputs=[values], device="cpu")
        self.assertIs(wp.get_module(__name__), original_module)
        np.testing.assert_allclose(values.numpy(), np.array([2.0, 3.0], dtype=np.float32))


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
