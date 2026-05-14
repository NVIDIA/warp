# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import warp
import warp as wp
from warp._src import build as _build_module
from warp._src import logger as _logger_module
from warp._src.logger import LoggerBasic, log_debug, log_error, log_warning


@wp.kernel
def _print_launches_test_kernel():
    pass


class _NoOpLogger:
    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message, category=None, stacklevel=1):
        pass

    def error(self, message):
        pass


class TestLogger(unittest.TestCase):
    def setUp(self):
        self._source_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._source_dir.cleanup)

    def _source_path(self, filename):
        return str(Path(self._source_dir.name) / filename)

    def _clear_deprecated_config_warnings(self):
        wp.config._deprecated_verbose_warning_seen = False
        wp.config._deprecated_quiet_warning_seen = False
        _logger_module._warnings_seen.clear()

    def test_log_level_constants(self):
        self.assertEqual(wp.LOG_DEBUG, 10)
        self.assertEqual(wp.LOG_INFO, 20)
        self.assertEqual(wp.LOG_WARNING, 30)
        self.assertEqual(wp.LOG_ERROR, 40)

    def test_logger_protocol_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            wp.Logger()

    def test_loggerkit_not_exported_from_warp_utils(self):
        self.assertFalse(hasattr(wp.utils, "LoggerKit"))

    def test_deprecated_config_verbose_read_warns_for_external_callers(self):
        self._clear_deprecated_config_warnings()
        namespace = {"wp": wp, "__name__": "external_app"}
        external_app_path = self._source_path("external_app.py")
        original_verbose_warnings = wp.config.verbose_warnings
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            wp.config.verbose_warnings = True
            with warnings.catch_warnings():
                warnings.simplefilter("always", DeprecationWarning)
                exec(
                    compile("value = wp.config.verbose\nagain = wp.config.verbose", external_app_path, "exec"),
                    namespace,
                )
            output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
            wp.config.verbose_warnings = original_verbose_warnings

        self.assertIsInstance(namespace["value"], bool)
        self.assertEqual(namespace["again"], namespace["value"])
        self.assertEqual(output.count("warp.config.verbose is deprecated"), 1)
        self.assertIn(external_app_path, output)

    def test_deprecated_config_quiet_assignment_warns_for_external_callers(self):
        self._clear_deprecated_config_warnings()
        original_quiet = wp.config.__dict__["quiet"]
        original_verbose_warnings = wp.config.verbose_warnings
        external_app_path = self._source_path("external_app.py")
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            wp.config.verbose_warnings = True
            namespace = {"wp": wp, "__name__": "external_app"}
            with warnings.catch_warnings():
                warnings.simplefilter("always", DeprecationWarning)
                exec(compile("wp.config.quiet = True", external_app_path, "exec"), namespace)
            output = sys.stderr.getvalue()

            self.assertTrue(wp.config.__dict__["quiet"])
            self.assertEqual(output.count("warp.config.quiet is deprecated"), 1)
            self.assertIn(external_app_path, output)
        finally:
            sys.stderr = old_stderr
            wp.config.verbose_warnings = original_verbose_warnings
            wp.config.__dict__["quiet"] = original_quiet

    def test_deprecated_config_access_does_not_warn_for_warp_callers(self):
        self._clear_deprecated_config_warnings()
        namespace = {"wp": wp, "__name__": "warp._src.fake_internal"}
        fake_internal_path = self._source_path("fake_internal.py")
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", DeprecationWarning)
            exec(
                compile(
                    "value = wp.config.verbose\nwp.config.quiet = wp.config.quiet",
                    fake_internal_path,
                    "exec",
                ),
                namespace,
            )

        self.assertIsInstance(namespace["value"], bool)
        self.assertEqual(recorded, [])

    def test_deprecated_config_access_routes_to_active_logger(self):
        class CaptureLogger:
            def __init__(self):
                self.warnings = []

            def debug(self, message):
                pass

            def info(self, message):
                pass

            def warning(self, message, category=None, stacklevel=1):
                self.warnings.append((message, category, stacklevel))

            def error(self, message):
                pass

        self._clear_deprecated_config_warnings()
        logger = CaptureLogger()
        original_logger = wp.get_logger()
        original_quiet = wp.config.__dict__["quiet"]
        external_app_path = self._source_path("external_app.py")
        try:
            wp.set_logger(logger)
            namespace = {"wp": wp, "__name__": "external_app"}
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", DeprecationWarning)
                exec(compile("wp.config.quiet = True", external_app_path, "exec"), namespace)
        finally:
            wp.config.__dict__["quiet"] = original_quiet
            wp.set_logger(original_logger)

        self.assertEqual(recorded, [])
        self.assertEqual(len(logger.warnings), 1)
        message, category, _stacklevel = logger.warnings[0]
        self.assertIn("warp.config.quiet is deprecated", message)
        self.assertEqual(category, DeprecationWarning)

    def test_deprecated_config_access_respects_log_level(self):
        self._clear_deprecated_config_warnings()
        original_level = wp.config.__dict__["log_level"]
        try:
            wp.config.__dict__["log_level"] = wp.LOG_ERROR
            namespace = {"wp": wp, "__name__": "external_app"}
            external_app_path = self._source_path("external_app.py")
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", DeprecationWarning)
                exec(compile("value = wp.config.verbose", external_app_path, "exec"), namespace)
        finally:
            wp.config.__dict__["log_level"] = original_level

        self.assertEqual(recorded, [])

    def test_set_logger_accepts_duck_typed_object(self):
        """Logger is a Protocol -- any object with the four methods works."""

        class DuckLogger:
            def debug(self, message):
                pass

            def info(self, message):
                pass

            def warning(self, message, category=None, stacklevel=1):
                pass

            def error(self, message):
                pass

        original = wp.get_logger()
        try:
            wp.set_logger(DuckLogger())
            self.assertIsInstance(wp.get_logger(), DuckLogger)
        finally:
            wp.set_logger(original)

    def test_set_logger_rejects_object_missing_methods(self):
        class NotALogger:
            def debug(self, message):
                pass

            # missing info, warning, error

        with self.assertRaises(TypeError):
            wp.set_logger(NotALogger())

    def test_basic_logger_debug_writes_to_stdout(self):
        logger = LoggerBasic()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            logger.debug("test debug msg")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertEqual(output, "test debug msg\n")

    def test_basic_logger_info_writes_to_stdout(self):
        logger = LoggerBasic()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            logger.info("test info msg")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertEqual(output, "test info msg\n")

    def test_basic_logger_error_writes_to_stderr(self):
        logger = LoggerBasic()
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            logger.error("something broke")
            output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
        self.assertEqual(output, "Warp Error: something broke\n")

    def test_basic_logger_warning_respects_filters(self):
        """GH-1315: user warning filters must not be overridden."""
        logger = LoggerBasic()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                logger.warning("old API", category=DeprecationWarning, stacklevel=1)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
        self.assertEqual(output, "", "DeprecationWarning should have been suppressed")

    def test_set_get_logger(self):
        original = wp.get_logger()
        try:
            self.assertIsInstance(original, LoggerBasic)
            logger = _NoOpLogger()
            wp.set_logger(logger)
            self.assertIs(wp.get_logger(), logger)
        finally:
            wp.set_logger(original)

    def test_set_logger_validates_type(self):
        with self.assertRaises(TypeError):
            wp.set_logger("not a logger")

    def test_set_logger_none_resets_to_basic_logger(self):
        original = wp.get_logger()
        try:
            wp.set_logger(_NoOpLogger())
            self.assertIsInstance(wp.get_logger(), _NoOpLogger)
            wp.set_logger(None)
            self.assertIsInstance(wp.get_logger(), LoggerBasic)
        finally:
            wp.set_logger(original)

    def test_scoped_logger_swaps_and_restores(self):
        original = wp.get_logger()
        logger = _NoOpLogger()
        with wp.ScopedLogger(logger) as scope:
            self.assertIs(scope.logger, logger)
            self.assertIs(wp.get_logger(), logger)
        self.assertIs(wp.get_logger(), original)

    def test_scoped_logger_restores_on_exception(self):
        original = wp.get_logger()
        with self.assertRaises(RuntimeError):
            with wp.ScopedLogger(_NoOpLogger()):
                raise RuntimeError("boom")
        self.assertIs(wp.get_logger(), original)

    def test_scoped_logger_none_uses_basic_logger(self):
        original = wp.get_logger()
        wp.set_logger(_NoOpLogger())
        try:
            with wp.ScopedLogger(None):
                self.assertIsInstance(wp.get_logger(), LoggerBasic)
        finally:
            wp.set_logger(original)

    def test_scoped_log_level_swaps_and_restores(self):
        original = wp.config.log_level
        try:
            wp.config.log_level = wp.LOG_INFO
            with wp.ScopedLogLevel(wp.LOG_ERROR):
                self.assertEqual(wp.config.log_level, wp.LOG_ERROR)
            self.assertEqual(wp.config.log_level, wp.LOG_INFO)
        finally:
            wp.config.log_level = original

    def test_scoped_log_level_restores_on_exception(self):
        original = wp.config.log_level
        try:
            wp.config.log_level = wp.LOG_INFO
            with self.assertRaises(RuntimeError):
                with wp.ScopedLogLevel(wp.LOG_ERROR):
                    raise RuntimeError("boom")
            self.assertEqual(wp.config.log_level, wp.LOG_INFO)
        finally:
            wp.config.log_level = original

    def test_log_debug_gated_by_level(self):
        original_level = warp.config.log_level
        original_verbose = warp.config.verbose
        warp.config.log_level = wp.LOG_WARNING
        warp.config.verbose = False
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            log_debug("should not appear")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            warp.config.log_level = original_level
            warp.config.verbose = original_verbose
        self.assertEqual(output, "")

    def test_log_debug_honors_deprecated_verbose_flag_at_call_time(self):
        original_level = warp.config.log_level
        original_verbose = warp.config.verbose
        warp.config.log_level = wp.LOG_WARNING
        warp.config.verbose = True
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            log_debug("debug via verbose")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            warp.config.log_level = original_level
            warp.config.verbose = original_verbose
        self.assertEqual(output, "debug via verbose\n")

    def test_cuda_build_honors_deprecated_verbose_flag_at_call_time(self):
        original_level = warp.config.__dict__["log_level"]
        original_verbose = warp.config.__dict__["verbose"]
        original_runtime = _build_module.warp._src.context.runtime
        captured = {}

        def compile_cuda(*args):
            captured["verbose"] = args[9]
            return 0

        try:
            warp.config.log_level = wp.LOG_WARNING
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warp.config.verbose = True
            _build_module.warp._src.context.runtime = SimpleNamespace(
                core=SimpleNamespace(wp_cuda_compile_program=compile_cuda)
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                cu_path = f"{tmpdir}/kernel.cu"
                output_path = f"{tmpdir}/kernel.ptx"
                with open(cu_path, "w") as cu_file:
                    cu_file.write("// empty test kernel\n")
                _build_module.build_cuda(cu_path, 80, output_path, pch_dir=None)
        finally:
            _build_module.warp._src.context.runtime = original_runtime
            warp.config.__dict__["log_level"] = original_level
            warp.config.__dict__["verbose"] = original_verbose

        self.assertTrue(captured["verbose"])

    def test_log_debug_emits_at_debug_level(self):
        original_level = warp.config.log_level
        warp.config.log_level = wp.LOG_DEBUG
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            log_debug("debug msg")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            warp.config.log_level = original_level
        self.assertEqual(output, "debug msg\n")

    def test_log_error_always_emits(self):
        """log_error ignores log_level -- errors are never suppressed."""
        original_level = warp.config.log_level
        warp.config.log_level = wp.LOG_ERROR + 10
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            log_error("critical failure")
            output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
            warp.config.log_level = original_level
        self.assertIn("critical failure", output)

    def test_log_warning_once_deduplicates(self):
        saved_warnings = _logger_module._warnings_seen.copy()
        _logger_module._warnings_seen.clear()
        with warnings.catch_warnings():
            warnings.resetwarnings()
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                log_warning("dup warning", category=UserWarning, once=True)
                log_warning("dup warning", category=UserWarning, once=True)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
                _logger_module._warnings_seen.clear()
                _logger_module._warnings_seen.update(saved_warnings)
        self.assertEqual(output.count("dup warning"), 1)

    def test_custom_logger_warning_stacklevel_points_to_external_caller(self):
        class WarningsLogger:
            def debug(self, message):
                pass

            def info(self, message):
                pass

            def warning(self, message, category=None, stacklevel=1):
                warnings.warn(message, category, stacklevel=stacklevel)

            def error(self, message):
                pass

        original = wp.get_logger()
        try:
            wp.set_logger(WarningsLogger())
            namespace = {"log_warning": log_warning, "UserWarning": UserWarning}
            warning_caller_path = self._source_path("external_warning_caller.py")
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", UserWarning)
                exec(
                    compile(
                        "log_warning('custom logger warning', category=UserWarning)",
                        warning_caller_path,
                        "exec",
                    ),
                    namespace,
                )
        finally:
            wp.set_logger(original)

        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].filename, warning_caller_path)

    def test_from_ptr_deprecation_warning_visible_to_default_filter(self):
        script = textwrap.dedent(
            """
            import warp as wp

            try:
                wp.from_ptr(0, 0, dtype=wp.float32)
            except TypeError:
                pass
            """
        )

        result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
        self.assertIn("Warp DeprecationWarning: This version of wp.from_ptr()", result.stderr)

    def test_print_launches_routes_to_active_logger(self):
        class CaptureLogger:
            def __init__(self):
                self.infos = []

            def debug(self, message):
                pass

            def info(self, message):
                self.infos.append(message)

            def warning(self, message, category=None, stacklevel=1):
                pass

            def error(self, message):
                pass

        logger = CaptureLogger()
        original_logger = wp.get_logger()
        original_print_launches = wp.config.print_launches
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            wp.set_logger(logger)
            wp.config.print_launches = True
            wp.launch(_print_launches_test_kernel, dim=0, device="cpu")
            stdout_output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            wp.config.print_launches = original_print_launches
            wp.set_logger(original_logger)

        self.assertEqual(stdout_output, "")
        self.assertTrue(any("kernel: _print_launches_test_kernel" in message for message in logger.infos))

    def test_log_warning_deprecation_warnings_deduplicate_without_once(self):
        """DeprecationWarnings are deduplicated even when ``once`` is not passed,
        matching the legacy ``warn()`` helper this replaced."""
        saved = _logger_module._warnings_seen.copy()
        _logger_module._warnings_seen.clear()
        with warnings.catch_warnings():
            warnings.resetwarnings()
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                log_warning("legacy api", category=DeprecationWarning)
                log_warning("legacy api", category=DeprecationWarning)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
                _logger_module._warnings_seen.clear()
                _logger_module._warnings_seen.update(saved)
        self.assertEqual(output.count("legacy api"), 1)

    def test_scoped_logger_exit_tolerates_mutated_saved_logger(self):
        """``ScopedLogger.__exit__`` must restore the saved logger without
        re-validating it; otherwise a TypeError would mask any in-flight
        exception propagating through the context."""

        class MutableLogger:
            def debug(self, message):
                pass

            def info(self, message):
                pass

            def warning(self, message, category=None, stacklevel=1):
                pass

            def error(self, message):
                pass

        original = wp.get_logger()
        outer = MutableLogger()
        wp.set_logger(outer)
        try:
            with wp.ScopedLogger(MutableLogger()):
                # Break the saved logger after entering the scope; __exit__
                # must still restore it without raising.
                outer.debug = None
            self.assertIs(wp.get_logger(), outer)
        finally:
            wp.set_logger(original)

    def test_gh1315_user_filters_respected(self):
        """Verify that warnings.filterwarnings('ignore') suppresses Warp warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                log_warning("deprecated thing", category=DeprecationWarning)
                output = sys.stderr.getvalue()
            finally:
                sys.stderr = old_stderr
        self.assertEqual(output, "", "DeprecationWarning should have been suppressed by user filter")


if __name__ == "__main__":
    unittest.main()
