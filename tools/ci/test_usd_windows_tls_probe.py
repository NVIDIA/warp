# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import usd_windows_tls_probe as probe

REPOSITORY_ROOT = Path(__file__).parents[2]


class FormatExitCodeTests(unittest.TestCase):
    def test_formats_windows_status_as_unsigned_hexadecimal(self):
        self.assertEqual(probe.format_exit_code(0xC0000005), "0xC0000005")
        self.assertEqual(probe.format_exit_code(-1073741819), "0xC0000005")


class ModuleFilteringTests(unittest.TestCase):
    def test_keeps_runtime_modules_relevant_to_usd_loading(self):
        modules = [
            r"C:\Windows\System32\kernel32.dll",
            r"C:\Windows\System32\msvcp140.dll",
            r"C:\venv\Lib\site-packages\torch\lib\torch_cpu.dll",
            r"C:\venv\Lib\site-packages\pxr\usd_ms.dll",
        ]

        self.assertEqual(
            probe.filter_relevant_module_paths(modules),
            [
                r"C:\Windows\System32\msvcp140.dll",
                r"C:\venv\Lib\site-packages\pxr\usd_ms.dll",
                r"C:\venv\Lib\site-packages\torch\lib\torch_cpu.dll",
            ],
        )


@unittest.skipUnless(os.name == "nt", "Windows API probe")
class WindowsApiTests(unittest.TestCase):
    def test_loaded_modules_include_the_python_runtime(self):
        modules = probe.loaded_module_paths()

        self.assertTrue(any(Path(path).name.lower().startswith("python") for path in modules))
        self.assertGreaterEqual(probe.native_thread_count(), 1)


class ChildProcessTests(unittest.TestCase):
    def test_nonzero_child_result_is_recorded_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            result = probe.run_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('child stdout'); print('child stderr', file=sys.stderr); sys.exit(7)",
                ],
                stdout_path=output_directory / "stdout.log",
                stderr_path=output_directory / "stderr.log",
                timeout_seconds=30,
            )

            self.assertEqual(result.return_code, 7)
            self.assertEqual(result.return_code_hex, "0x00000007")
            self.assertFalse(result.timed_out)
            self.assertEqual((output_directory / "stdout.log").read_text(encoding="utf-8").strip(), "child stdout")
            self.assertEqual((output_directory / "stderr.log").read_text(encoding="utf-8").strip(), "child stderr")

    def test_late_import_creates_pressure_threads_after_cuda_warmup(self):
        events = []

        @contextlib.contextmanager
        def fake_hold_threads(count):
            events.append(("hold-enter", count))
            yield
            events.append(("hold-exit", count))

        with (
            mock.patch.object(probe, "emit_event"),
            mock.patch.object(probe, "hold_threads", fake_hold_threads),
            mock.patch.object(probe, "warm_cuda_runtimes", side_effect=lambda: events.append("warmup") or (1, 2)),
            mock.patch.object(probe, "import_tf", side_effect=lambda: events.append("tf")),
        ):
            return_code = probe.run_child("late-tf", held_thread_count=16, attempt=1)

        self.assertEqual(return_code, 0)
        self.assertEqual(events, ["warmup", ("hold-enter", 16), "tf", ("hold-exit", 16)])

    def test_import_order_case_runs_only_its_selected_preload(self):
        events = []

        with (
            mock.patch.object(probe, "emit_event"),
            mock.patch.object(probe, "warm_msvc_runtime", side_effect=lambda: events.append("msvc") or 1),
            mock.patch.object(probe, "import_blosc", side_effect=lambda: events.append("blosc") or 2),
            mock.patch.object(probe, "import_warp", side_effect=lambda: events.append("warp") or 3),
            mock.patch.object(probe, "warm_warp_cuda", side_effect=lambda: events.append("warp-cuda") or 4),
            mock.patch.object(probe, "import_torch", side_effect=lambda: events.append("torch") or 5),
            mock.patch.object(probe, "warm_torch_cuda", side_effect=lambda: events.append("torch-cuda") or 6),
            mock.patch.object(probe, "import_tf", side_effect=lambda: events.append("tf")),
        ):
            return_code = probe.run_child("after-torch-import-tf", held_thread_count=0, attempt=1)

        self.assertEqual(return_code, 0)
        self.assertEqual(events, ["torch", "tf"])

    def test_pxr_path_case_extends_the_search_path_before_tf(self):
        events = []

        with (
            mock.patch.object(probe, "emit_event"),
            mock.patch.object(probe, "extend_pxr_dll_path", side_effect=lambda: events.append("pxr-path") or 1),
            mock.patch.object(probe, "import_tf", side_effect=lambda: events.append("tf")),
        ):
            return_code = probe.run_child("after-pxr-dll-path-tf", held_thread_count=0, attempt=1)

        self.assertEqual(return_code, 0)
        self.assertEqual(events, ["pxr-path", "tf"])

    def test_procdump_wraps_child_and_uses_a_unique_dump_directory(self):
        output_directory = Path(r"C:\artifacts\early-tf")
        command = probe.build_child_command(
            case="early-tf",
            held_thread_count=0,
            attempt=3,
            output_directory=output_directory,
            procdump_path=Path(r"C:\tools\procdump64.exe"),
        )

        self.assertEqual(command[0], r"C:\tools\procdump64.exe")
        self.assertIn("-e", command)
        self.assertIn("C0000005", command)
        self.assertIn(str(output_directory / "dumps" / "early-tf-threads-000-attempt-003"), command)
        self.assertIn("--child", command)


class EagerImportTests(unittest.TestCase):
    def test_sitecustomize_imports_tf_before_the_python_body(self):
        eager_directory = Path(__file__).parent / "usd_eager_import"
        with tempfile.TemporaryDirectory() as directory:
            fake_package_root = Path(directory)
            pxr_directory = fake_package_root / "pxr"
            pxr_directory.mkdir()
            (pxr_directory / "__init__.py").write_text("from . import Tf\n", encoding="utf-8")
            (pxr_directory / "Tf.py").write_text("VALUE = 'loaded'\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join((str(eager_directory), str(fake_package_root)))

            result = subprocess.run(
                [sys.executable, "-c", "print('python-body')"],
                capture_output=True,
                check=False,
                encoding="utf-8",
                env=environment,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertIn('"kind": "usd-eager-import"', lines[0])
            self.assertEqual(lines[-1], "python-body")


class DiagnosticWorkflowTests(unittest.TestCase):
    def test_diagnostic_workflow_runs_only_msvc_resolution_probe(self):
        workflow_path = REPOSITORY_ROOT / ".github" / "workflows" / "usd-windows-tls-diagnostic.yml"
        self.assertTrue(workflow_path.exists(), f"Missing {workflow_path}")
        workflow = workflow_path.read_text(encoding="utf-8")

        self.assertIn("runs-on: windows-amd64-gpu-rtxpro6000-latest-1", workflow)
        self.assertIn("Run MSVC resolution probes", workflow)
        self.assertIn("uv venv .probe-venv", workflow)
        self.assertIn("uv pip install --python $probePython", workflow)
        self.assertIn("& $probePython tools/ci/usd_windows_tls_probe.py", workflow)
        self.assertIn("blosc==1.11.4", workflow)
        self.assertIn("'after-pxr-dll-path-tf'", workflow)
        self.assertIn("Report MSVC runtime candidates", workflow)
        self.assertNotIn("build-warp-windows-cuda13", workflow)
        self.assertNotIn("--extra dev", workflow)
        self.assertNotIn("procdump64.exe", workflow)
        self.assertIn("if: always()", workflow)

    def test_pull_request_workflow_routes_only_to_the_diagnostic_workflow(self):
        workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "pr.yml").read_text(encoding="utf-8")

        self.assertIn("uses: ./.github/workflows/usd-windows-tls-diagnostic.yml", workflow)
        self.assertNotIn("\n  cppcheck:", workflow)
        self.assertNotIn("\n  linkcheck:", workflow)


if __name__ == "__main__":
    unittest.main()
