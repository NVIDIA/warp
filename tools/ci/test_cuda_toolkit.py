# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import unittest
import urllib.error
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import cuda_toolkit as ctk


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def archive_record(
    payload: bytes,
    component: str = "cuda_nvcc",
    relative_path: str | None = None,
) -> ctk.ArchiveRecord:
    return ctk.ArchiveRecord(
        component=component,
        relative_path=relative_path or f"{component}/linux-x86_64/{component}-archive.tar.xz",
        size=len(payload),
        sha256=sha256(payload),
    )


def opener_returning(payload: bytes):
    def open_response(request: object, timeout: int) -> io.BytesIO:
        return io.BytesIO(payload)

    return open_response


class SequencedOpener:
    def __init__(self, *results: bytes | Exception):
        self.results = list(results)
        self.calls = 0

    def __call__(self, request: object, timeout: int) -> io.BytesIO:
        result = self.results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(result)


def tar_archive(root: str, entries: dict[str, bytes | tuple[str, str]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:xz") as archive:
        for name, value in entries.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            if isinstance(value, tuple):
                info.type = tarfile.SYMTYPE
                info.linkname = value[1]
                archive.addfile(info)
            else:
                info.size = len(value)
                info.mode = 0o755 if name.startswith("bin/") else 0o644
                archive.addfile(info, io.BytesIO(value))
    return output.getvalue()


def zip_archive(root: str, entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in entries.items():
            archive.writestr(f"{root}/{name}", value)
    return output.getvalue()


def make_toolkit(root: Path, platform: str, exit_code: int = 0) -> Path:
    executable = "nvcc.exe" if platform.startswith("windows-") else "nvcc"
    toolkit = root / "toolkit"
    (toolkit / "include").mkdir(parents=True)
    (toolkit / "bin").mkdir()
    nvcc = toolkit / "bin" / executable
    nvcc.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
    nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)
    return toolkit


class ToolkitTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)


class ConfigTests(ToolkitTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.requirements_path = self.directory / "requirements.json"
        self.lock_path = self.directory / "lock.json"
        self.requirements_data = {
            "schema_version": 1,
            "component_sets": {
                "build": ["cuda_nvcc", "cuda_cudart"],
            },
            "releases": {
                "13.0.2": {
                    "component_set": "build",
                    "platforms": ["linux-x86_64"],
                }
            },
        }
        self.lock_data = {
            "schema_version": 1,
            "bundles": {
                "13.0.2": {
                    "linux-x86_64": [
                        {
                            "component": "cuda_nvcc",
                            "relative_path": "cuda_nvcc/linux-x86_64/nvcc.tar.xz",
                            "size": 4,
                            "sha256": sha256(b"nvcc"),
                        },
                        {
                            "component": "cuda_cudart",
                            "relative_path": "cuda_cudart/linux-x86_64/cudart.tar.xz",
                            "size": 6,
                            "sha256": sha256(b"cudart"),
                        },
                    ]
                }
            },
        }
        write_json(self.requirements_path, self.requirements_data)
        write_json(self.lock_path, self.lock_data)

    def test_generate_lock_uses_manifest_keys(self) -> None:
        """Generate lock records from NVIDIA manifest keys."""
        requirements = ctk.load_requirements(self.requirements_path)
        manifests = {
            "13.0.2": {
                "cuda_nvcc": {
                    "linux-x86_64": {
                        "relative_path": "cuda_nvcc/linux-x86_64/nvcc.tar.xz",
                        "size": "4",
                        "sha256": sha256(b"nvcc"),
                    }
                },
                "cuda_cudart": {
                    "linux-x86_64": {
                        "relative_path": "cuda_cudart/linux-x86_64/cudart.tar.xz",
                        "size": "6",
                        "sha256": sha256(b"cudart"),
                    }
                },
            }
        }

        lock = ctk.generate_lock(requirements, manifests)

        self.assertEqual(
            lock.bundles[("13.0.2", "linux-x86_64")].archives,
            (
                ctk.ArchiveRecord(
                    "cuda_nvcc",
                    "cuda_nvcc/linux-x86_64/nvcc.tar.xz",
                    4,
                    sha256(b"nvcc"),
                ),
                ctk.ArchiveRecord(
                    "cuda_cudart",
                    "cuda_cudart/linux-x86_64/cudart.tar.xz",
                    6,
                    sha256(b"cudart"),
                ),
            ),
        )

    def test_resolve_outputs_assembled_cache(self) -> None:
        """Resolve an assembled-toolkit cache path."""
        output = self.directory / "github-output"
        with (
            mock.patch.object(ctk, "REQUIREMENTS_PATH", self.requirements_path),
            mock.patch.object(ctk, "LOCK_PATH", self.lock_path),
        ):
            result = ctk.main(
                [
                    "resolve",
                    "--version",
                    "13.0.2",
                    "--platform",
                    "linux-x86_64",
                    "--github-output",
                    str(output),
                    "--runner-temp",
                    str(self.directory),
                ]
            )

        self.assertEqual(result, 0)
        values = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
        self.assertRegex(
            values["cache-key"],
            r"^warp-cuda-toolkit-v1-13\.0\.2-linux-x86_64-[0-9a-f]{64}$",
        )
        self.assertEqual(
            Path(values["cuda-path"]).parent,
            self.directory / "warp-cuda-toolkits",
        )
        self.assertNotIn("archive-dir", values)

    def test_reject_unknown_release(self) -> None:
        """Reject an unconfigured CUDA Toolkit release."""
        requirements = ctk.load_requirements(self.requirements_path)
        lock = ctk.load_lock(self.lock_path)

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "Unknown"):
            ctk.resolve_bundle(requirements, lock, "13.1.0", "linux-x86_64")

    def test_reject_unsupported_platform(self) -> None:
        """Reject an unsupported release platform."""
        requirements = ctk.load_requirements(self.requirements_path)
        lock = ctk.load_lock(self.lock_path)

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "does not support"):
            ctk.resolve_bundle(requirements, lock, "13.0.2", "linux-sbsa")

    def test_reject_component_order_mismatch(self) -> None:
        """Reject a lock with mismatched component order."""
        archives = self.lock_data["bundles"]["13.0.2"]["linux-x86_64"]
        archives.reverse()
        write_json(self.lock_path, self.lock_data)
        requirements = ctk.load_requirements(self.requirements_path)
        lock = ctk.load_lock(self.lock_path)

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "components"):
            ctk.resolve_bundle(requirements, lock, "13.0.2", "linux-x86_64")

    def test_reject_extra_lock_bundle(self) -> None:
        """Reject a lock with an unconfigured bundle."""
        self.lock_data["bundles"]["13.1.0"] = {"linux-x86_64": self.lock_data["bundles"]["13.0.2"]["linux-x86_64"]}
        write_json(self.lock_path, self.lock_data)
        requirements = ctk.load_requirements(self.requirements_path)
        lock = ctk.load_lock(self.lock_path)

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "extra bundles"):
            ctk.validate_lock(requirements, lock)


class DownloadTests(ToolkitTestCase):
    def test_verify_download(self) -> None:
        """Verify a downloaded NVIDIA archive."""
        payload = b"locked archive"

        path = ctk.download_archive(
            archive_record(payload),
            self.directory,
            opener=opener_returning(payload),
        )

        self.assertEqual(path.read_bytes(), payload)

    def test_reject_size_mismatch(self) -> None:
        """Reject a downloaded archive with the wrong size."""
        record = replace(archive_record(b"expected"), size=99)

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "size"):
            ctk.download_archive(
                record,
                self.directory,
                opener=opener_returning(b"expected"),
            )

    def test_reject_digest_mismatch(self) -> None:
        """Reject a downloaded archive with the wrong digest."""
        record = archive_record(b"expected")

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "SHA-256"):
            ctk.download_archive(
                record,
                self.directory,
                opener=opener_returning(b"mismatch"),
            )

    def test_retry_transient_download(self) -> None:
        """Retry a transient NVIDIA download failure."""
        opener = SequencedOpener(
            urllib.error.URLError("temporary"),
            b"locked archive",
        )

        ctk.download_archive(
            archive_record(b"locked archive"),
            self.directory,
            opener=opener,
            sleep=lambda _: None,
        )

        self.assertEqual(opener.calls, 2)


class InstallTests(ToolkitTestCase):
    def test_assemble_components_in_order(self) -> None:
        """Assemble trusted components in lock order."""
        first = tar_archive(
            "cuda_nvcc",
            {
                "include/cuda.h": b"header",
                "lib/libfirst.a": b"first",
                # The second component must replace this symlink with a file.
                "shared/value": ("symlink", "old"),
            },
        )
        second = zip_archive(
            "cuda_cudart",
            {
                "lib/libsecond.a": b"second",
                "shared/value": b"new",
            },
        )
        bundle = ctk.Bundle(
            "13.0.2",
            "linux-x86_64",
            (
                archive_record(first),
                archive_record(
                    second,
                    "cuda_cudart",
                    "cuda_cudart/linux-x86_64/cudart.zip",
                ),
            ),
        )
        destination = self.directory / "cuda"

        result = ctk.install_bundle(
            bundle,
            destination,
            opener=SequencedOpener(first, second),
            sleep=lambda _: None,
        )

        self.assertEqual(result, destination.resolve())
        self.assertEqual((destination / "shared/value").read_bytes(), b"new")
        self.assertEqual(
            sorted(path.name for path in (destination / "lib64").iterdir()),
            ["libfirst.a", "libsecond.a"],
        )
        self.assertFalse((destination / "lib").exists())

    def test_replace_partial_destination_after_assembly(self) -> None:
        """Replace a partial destination after successful assembly."""
        payload = tar_archive("cuda_nvcc", {"include/cuda.h": b"header"})
        destination = self.directory / "cuda"
        destination.mkdir()
        (destination / "partial").write_text("old", encoding="utf-8")

        ctk.install_bundle(
            ctk.Bundle(
                "13.0.2",
                "linux-x86_64",
                (archive_record(payload),),
            ),
            destination,
            opener=opener_returning(payload),
        )

        self.assertFalse((destination / "partial").exists())
        self.assertEqual(
            (destination / "include/cuda.h").read_bytes(),
            b"header",
        )

    def test_preserve_destination_after_failed_assembly(self) -> None:
        """Preserve an existing destination after failed assembly."""
        payload = tar_archive("cuda_nvcc", {"include/cuda.h": b"header"})
        destination = self.directory / "cuda"
        destination.mkdir()
        marker = destination / "partial"
        marker.write_text("old", encoding="utf-8")

        with self.assertRaises(ctk.ToolkitConfigError):
            ctk.install_bundle(
                ctk.Bundle(
                    "13.0.2",
                    "linux-x86_64",
                    (archive_record(payload),),
                ),
                destination,
                opener=opener_returning(b"corrupt"),
            )

        self.assertEqual(marker.read_text(encoding="utf-8"), "old")


class PackageTests(ToolkitTestCase):
    def test_reject_mismatched_package_without_replacement(self) -> None:
        """Reject a mismatched package without replacing the destination."""
        source = make_toolkit(self.directory / "source", "linux-x86_64")
        (source / "payload").write_text("toolkit", encoding="utf-8")
        bundle = ctk.Bundle(
            "13.0.2",
            "linux-x86_64",
            (archive_record(b"locked"),),
        )
        package = self.directory / "cuda-toolkit.tar.xz"
        ctk.pack_toolkit(bundle, source, package)
        destination = self.directory / "destination"
        destination.mkdir()
        marker = destination / "marker"
        marker.write_text("existing", encoding="utf-8")

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "metadata"):
            ctk.unpack_toolkit(
                replace(bundle, version="13.0.3"),
                package,
                destination,
            )

        self.assertEqual(marker.read_text(encoding="utf-8"), "existing")


class ActivationTests(ToolkitTestCase):
    def test_activate_linux_toolkit(self) -> None:
        """Activate a Linux CUDA Toolkit."""
        toolkit = make_toolkit(self.directory, "linux-x86_64")
        github_env = self.directory / "github-env"
        github_path = self.directory / "github-path"

        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": ""}):
            ctk.activate_toolkit(
                toolkit,
                "linux-x86_64",
                github_env,
                github_path,
            )

        self.assertEqual(
            github_env.read_text(encoding="utf-8"),
            (
                f"WARP_CUDA_PATH={toolkit.resolve()}\n"
                f"CUDA_HOME={toolkit.resolve()}\n"
                f"CUDA_PATH={toolkit.resolve()}\n"
                f"LD_LIBRARY_PATH={toolkit.resolve() / 'lib64'}\n"
            ),
        )
        self.assertEqual(
            github_path.read_text(encoding="utf-8"),
            f"{toolkit.resolve() / 'bin'}\n",
        )

    def test_activate_windows_toolkit(self) -> None:
        """Activate a Windows CUDA Toolkit."""
        toolkit = make_toolkit(self.directory, "windows-x86_64")
        github_env = self.directory / "github-env"
        github_path = self.directory / "github-path"

        ctk.activate_toolkit(
            toolkit,
            "windows-x86_64",
            github_env,
            github_path,
        )

        self.assertNotIn(
            "LD_LIBRARY_PATH",
            github_env.read_text(encoding="utf-8"),
        )

    def test_reject_missing_include(self) -> None:
        """Reject a Toolkit without headers."""
        toolkit = make_toolkit(self.directory, "linux-x86_64")
        (toolkit / "include").rmdir()

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "include"):
            ctk.activate_toolkit(toolkit, "linux-x86_64")

    def test_reject_missing_nvcc(self) -> None:
        """Reject a Toolkit without the CUDA compiler."""
        toolkit = make_toolkit(self.directory, "linux-x86_64")
        (toolkit / "bin/nvcc").unlink()

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "nvcc"):
            ctk.activate_toolkit(toolkit, "linux-x86_64")

    def test_reject_nonfunctional_nvcc(self) -> None:
        """Reject a Toolkit with a failing compiler."""
        toolkit = make_toolkit(self.directory, "linux-x86_64", exit_code=1)

        with self.assertRaisesRegex(ctk.ToolkitConfigError, "status 1"):
            ctk.activate_toolkit(toolkit, "linux-x86_64")


if __name__ == "__main__":
    unittest.main()
