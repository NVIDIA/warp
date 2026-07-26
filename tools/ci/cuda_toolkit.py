# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage the locked CUDA Toolkits used to build Warp in GitHub CI.

The ``setup-warp-cuda`` action restores a completed Toolkit directory from
GitHub's cache. On a miss, this helper downloads the configured NVIDIA
redistributable archives, verifies their locked sizes and SHA-256 digests,
and assembles them in requirements order. NVIDIA archives are trusted inputs;
the checks detect transport corruption rather than hostile archive content.

``cuda_toolkit_requirements.json`` lists each supported CUDA release,
NVIDIA manifest platform key, and ordered component set.
``cuda_toolkit_lock.json`` records the exact NVIDIA archive metadata.

To add or change a Toolkit:

1. Edit ``cuda_toolkit_requirements.json`` using NVIDIA manifest keys.
2. Refresh and validate the lock from the repository root::

       uv run --no-project python tools/ci/cuda_toolkit.py update-lock
       uv run --no-project python tools/ci/cuda_toolkit.py validate-lock

3. Review both JSON changes and open a normal pull request. For a forked
   contribution, an NVIDIA engineer must approve the exact commit so the
   copy-PR service mirrors it to ``pull-request/<number>``.
4. Rerun the mirrored pull request to exercise cache creation. After merge,
   default-branch CI can create the same cache if it is still absent.

To exchange an assembled Toolkit through a generic package registry::

    uv run --no-project python tools/ci/cuda_toolkit.py pack \
        --version VERSION --platform PLATFORM \
        --cuda-path CUDA_PATH --archive PACKAGE.tar.xz
    uv run --no-project python tools/ci/cuda_toolkit.py unpack \
        --version VERSION --platform PLATFORM \
        --archive PACKAGE.tar.xz --cuda-path CUDA_PATH

Registry upload, download, and environment export remain the CI workflow's
responsibility.

Fork workflows cannot read or write these caches. The action's ``cache-read``
and ``cache-write`` inputs carry repository workflow authorization; this
helper does not decide whether a caller is trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

LOCK_SCHEMA_VERSION = 1
INSTALL_SCHEMA_VERSION = 1
PACKAGE_SCHEMA_VERSION = 1
PACKAGE_ROOT = "cuda-toolkit"
PACKAGE_METADATA_NAME = ".warp-cuda-toolkit.json"
REQUIREMENTS_PATH = Path(__file__).with_name("cuda_toolkit_requirements.json")
LOCK_PATH = Path(__file__).with_name("cuda_toolkit_lock.json")
REDIST_BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
NVCC_TIMEOUT_SECONDS = 60


class ToolkitConfigError(ValueError):
    """Report invalid CUDA Toolkit configuration or installation data."""


@dataclass(frozen=True)
class ReleaseRequirements:
    component_set: str
    platforms: tuple[str, ...]


@dataclass(frozen=True)
class ToolkitRequirements:
    component_sets: dict[str, tuple[str, ...]]
    releases: dict[str, ReleaseRequirements]


@dataclass(frozen=True)
class ArchiveRecord:
    component: str
    relative_path: str
    size: int
    sha256: str

    def digest_value(self) -> dict[str, object]:
        return {
            "component": self.component,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class Bundle:
    version: str
    platform: str
    archives: tuple[ArchiveRecord, ...]

    @property
    def digest(self) -> str:
        payload = {
            "archives": [archive.digest_value() for archive in self.archives],
            "installation_schema": INSTALL_SCHEMA_VERSION,
            "platform": self.platform,
            "version": self.version,
        }
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cache_key(self) -> str:
        return f"warp-cuda-toolkit-v{INSTALL_SCHEMA_VERSION}-{self.version}-{self.platform}-{self.digest}"


@dataclass(frozen=True)
class ToolkitLock:
    bundles: dict[tuple[str, str], Bundle]


def _error(context: str, message: str) -> ToolkitConfigError:
    return ToolkitConfigError(f"Invalid CUDA Toolkit {context}: {message}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _error(context, "expected an object with string keys")
    return value


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        return _object(
            json.loads(path.read_text(encoding="utf-8")),
            context,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise _error(context, f"could not load {path}: {error}") from error


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise _error(context, f"missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise _error(context, f"unknown keys: {', '.join(sorted(unknown))}")


def _name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(context, "expected a nonempty string")
    return value


def _name_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _error(context, "expected a nonempty list")
    names = tuple(_name(item, context) for item in value)
    if len(names) != len(set(names)):
        raise _error(context, "contains duplicate names")
    return names


def _schema_version(value: Any, context: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != LOCK_SCHEMA_VERSION:
        raise _error(context, f"unsupported schema version: {value!r}")


def _relative_path(value: Any, context: str) -> str:
    text = _name(value, context)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise _error(context, "expected a normalized relative path")
    if not text.endswith((".tar.xz", ".zip")):
        raise _error(context, "expected a .tar.xz or .zip archive")
    return text


def _sha256(value: Any, context: str) -> str:
    digest = _name(value, context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise _error(
            context,
            "expected a 64-character lowercase SHA-256 digest",
        )
    return digest


def load_requirements(path: Path) -> ToolkitRequirements:
    """Load CUDA Toolkit requirements from ``path``."""
    data = _load_json(path, "requirements")
    _require_keys(
        data,
        {"schema_version", "component_sets", "releases"},
        "requirements",
    )
    _schema_version(data["schema_version"], "requirements")

    component_data = _object(
        data["component_sets"],
        "requirements component_sets",
    )
    if not component_data:
        raise _error(
            "requirements component_sets",
            "expected a nonempty object",
        )
    component_sets = {
        _name(name, "requirements component set name"): _name_list(
            components,
            f"requirements component set {name!r}",
        )
        for name, components in component_data.items()
    }

    release_data = _object(data["releases"], "requirements releases")
    if not release_data:
        raise _error("requirements releases", "expected a nonempty object")
    releases: dict[str, ReleaseRequirements] = {}
    for raw_version, raw_release in release_data.items():
        version = _name(raw_version, "requirements release version")
        release = _object(
            raw_release,
            f"requirements release {version!r}",
        )
        _require_keys(
            release,
            {"component_set", "platforms"},
            f"requirements release {version!r}",
        )
        component_set = _name(
            release["component_set"],
            f"requirements release {version!r} component_set",
        )
        if component_set not in component_sets:
            raise _error(
                f"requirements release {version!r}",
                f"unknown component set {component_set!r}",
            )
        releases[version] = ReleaseRequirements(
            component_set,
            _name_list(
                release["platforms"],
                f"requirements release {version!r} platforms",
            ),
        )
    return ToolkitRequirements(component_sets, releases)


def _archive_record(value: Any, context: str) -> ArchiveRecord:
    record = _object(value, context)
    _require_keys(
        record,
        {"component", "relative_path", "size", "sha256"},
        context,
    )
    size = record["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise _error(context, "expected a nonnegative integer size")
    return ArchiveRecord(
        component=_name(record["component"], f"{context} component"),
        relative_path=_relative_path(
            record["relative_path"],
            f"{context} relative_path",
        ),
        size=size,
        sha256=_sha256(record["sha256"], f"{context} sha256"),
    )


def load_lock(path: Path) -> ToolkitLock:
    """Load the CUDA Toolkit lock from ``path``."""
    data = _load_json(path, "lock")
    _require_keys(data, {"schema_version", "bundles"}, "lock")
    _schema_version(data["schema_version"], "lock")

    bundle_data = _object(data["bundles"], "lock bundles")
    bundles: dict[tuple[str, str], Bundle] = {}
    for raw_version, raw_platforms in bundle_data.items():
        version = _name(raw_version, "lock bundle version")
        platforms = _object(
            raw_platforms,
            f"lock bundle {version!r}",
        )
        for raw_platform, raw_archives in platforms.items():
            platform = _name(
                raw_platform,
                f"lock bundle {version!r} platform",
            )
            if not isinstance(raw_archives, list) or not raw_archives:
                raise _error(
                    f"lock bundle {version!r}/{platform!r}",
                    "expected a nonempty archive list",
                )
            archives = tuple(
                _archive_record(
                    raw_archive,
                    (f"lock bundle {version!r}/{platform!r} archive {index}"),
                )
                for index, raw_archive in enumerate(raw_archives)
            )
            components = tuple(archive.component for archive in archives)
            if len(components) != len(set(components)):
                raise _error(
                    f"lock bundle {version!r}/{platform!r}",
                    "contains duplicate components",
                )
            bundles[(version, platform)] = Bundle(
                version,
                platform,
                archives,
            )
    return ToolkitLock(bundles)


def resolve_bundle(
    requirements: ToolkitRequirements,
    lock: ToolkitLock,
    version: str,
    platform: str,
) -> Bundle:
    """Resolve a configured Toolkit bundle."""
    release = requirements.releases.get(version)
    if release is None:
        raise ToolkitConfigError(f"Unknown CUDA Toolkit release {version!r}")
    if platform not in release.platforms:
        raise ToolkitConfigError(f"CUDA Toolkit release {version!r} does not support platform {platform!r}")
    bundle = lock.bundles.get((version, platform))
    if bundle is None:
        raise ToolkitConfigError(f"Missing CUDA Toolkit lock bundle for {version!r}/{platform!r}")
    expected = requirements.component_sets[release.component_set]
    actual = tuple(archive.component for archive in bundle.archives)
    if actual != expected:
        raise ToolkitConfigError(
            f"CUDA Toolkit lock bundle for {version!r}/{platform!r} has components {actual!r}; expected {expected!r}"
        )
    return bundle


def validate_lock(
    requirements: ToolkitRequirements,
    lock: ToolkitLock,
) -> int:
    """Validate all configured Toolkit bundles."""
    expected = {
        (version, platform) for version, release in requirements.releases.items() for platform in release.platforms
    }
    missing = expected - set(lock.bundles)
    extra = set(lock.bundles) - expected
    if missing:
        raise ToolkitConfigError(f"CUDA Toolkit lock is missing bundles: {sorted(missing)!r}")
    if extra:
        raise ToolkitConfigError(f"CUDA Toolkit lock has extra bundles: {sorted(extra)!r}")
    return sum(len(resolve_bundle(requirements, lock, *key).archives) for key in sorted(expected))


def _manifest_archive(
    manifest: Mapping[str, Any],
    version: str,
    platform: str,
    component: str,
) -> ArchiveRecord:
    component_data = manifest.get(component)
    if not isinstance(component_data, Mapping):
        raise ToolkitConfigError(f"NVIDIA manifest for {version} does not provide {component!r}")
    archive = component_data.get(platform)
    if not isinstance(archive, Mapping):
        raise ToolkitConfigError(f"NVIDIA manifest for {version} does not provide {component!r} for {platform!r}")
    context = f"NVIDIA manifest {version!r} {component!r}/{platform!r}"
    try:
        relative_path = _relative_path(
            archive["relative_path"],
            f"{context} relative_path",
        )
        raw_size = archive["size"]
        digest = _sha256(archive["sha256"], f"{context} sha256")
    except KeyError as error:
        raise ToolkitConfigError(f"{context} is missing required field {error.args[0]!r}") from error
    if not isinstance(raw_size, str):
        raise ToolkitConfigError(f"{context} size must be a base-10 integer")
    try:
        size = int(raw_size, 10)
    except ValueError as error:
        raise ToolkitConfigError(f"{context} size must be a base-10 integer") from error
    if size < 0:
        raise ToolkitConfigError(f"{context} size must be nonnegative")
    return ArchiveRecord(component, relative_path, size, digest)


def generate_lock(
    requirements: ToolkitRequirements,
    manifests: Mapping[str, Mapping[str, Any]],
) -> ToolkitLock:
    """Generate a lock from NVIDIA release manifests."""
    bundles: dict[tuple[str, str], Bundle] = {}
    for version, release in requirements.releases.items():
        manifest = manifests.get(version)
        if not isinstance(manifest, Mapping):
            raise ToolkitConfigError(f"Missing NVIDIA manifest for CUDA Toolkit release {version!r}")
        components = requirements.component_sets[release.component_set]
        for platform in release.platforms:
            archives = tuple(
                _manifest_archive(
                    manifest,
                    version,
                    platform,
                    component,
                )
                for component in components
            )
            bundles[(version, platform)] = Bundle(
                version,
                platform,
                archives,
            )
    return ToolkitLock(bundles)


def _lock_data(lock: ToolkitLock) -> dict[str, object]:
    bundles: dict[str, dict[str, list[dict[str, object]]]] = {}
    for (version, platform), bundle in sorted(lock.bundles.items()):
        bundles.setdefault(version, {})[platform] = [archive.digest_value() for archive in bundle.archives]
    return {"schema_version": LOCK_SCHEMA_VERSION, "bundles": bundles}


def write_lock(lock: ToolkitLock, path: Path) -> None:
    """Write ``lock`` atomically and deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        json.dumps(
            _lock_data(lock),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(contents)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def load_redist_manifest(version: str) -> dict[str, Any]:
    """Download one NVIDIA redistributable manifest."""
    url = f"{REDIST_BASE_URL}/redistrib_{version}.json"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NVIDIA-Warp-CI"},
    )
    with urllib.request.urlopen(
        request,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ToolkitConfigError(f"NVIDIA manifest for {version} is not an object")
    return value


def update_lock(
    requirements_path: Path = REQUIREMENTS_PATH,
    lock_path: Path = LOCK_PATH,
) -> ToolkitLock:
    """Refresh the lock from NVIDIA redistributable manifests."""
    requirements = load_requirements(requirements_path)
    manifests = {version: load_redist_manifest(version) for version in requirements.releases}
    lock = generate_lock(requirements, manifests)
    validate_lock(requirements, lock)
    write_lock(lock, lock_path)
    return lock


def download_archive(
    record: ArchiveRecord,
    directory: Path,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download and verify one locked NVIDIA archive."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / PurePosixPath(record.relative_path).name
    url = f"{REDIST_BASE_URL}/{urllib.parse.quote(record.relative_path, safe='/')}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NVIDIA-Warp-CI"},
    )
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            digest = hashlib.sha256()
            size = 0
            with (
                opener(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
                destination.open("wb") as output,
            ):
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    size += len(chunk)
                    if size > record.size:
                        raise ToolkitConfigError(
                            f"Downloaded CUDA Toolkit archive "
                            f"{record.relative_path!r} exceeds locked size "
                            f"{record.size}"
                        )
                    digest.update(chunk)
                    output.write(chunk)
            if size != record.size:
                raise ToolkitConfigError(
                    f"Downloaded CUDA Toolkit archive {record.relative_path!r} has size {size}; expected {record.size}"
                )
            if digest.hexdigest() != record.sha256:
                raise ToolkitConfigError(
                    f"Downloaded CUDA Toolkit archive {record.relative_path!r} has the wrong SHA-256"
                )
            return destination
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt + 1 == DOWNLOAD_ATTEMPTS:
                raise ToolkitConfigError(
                    f"Could not download CUDA Toolkit archive {record.relative_path!r}: {error}"
                ) from error
            sleep(2**attempt)
    raise AssertionError("unreachable download retry state")


def extract_archive(archive: Path, destination: Path) -> Path:
    """Extract one trusted NVIDIA archive."""
    destination.mkdir(parents=True)
    try:
        if archive.name.endswith(".tar.xz"):
            with tarfile.open(archive, "r:xz") as stream:
                stream.extractall(destination, filter="data")
        elif archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as stream:
                stream.extractall(destination)
        else:
            raise ToolkitConfigError(f"Unsupported CUDA Toolkit archive {archive}")
    except (
        tarfile.TarError,
        zipfile.BadZipFile,
        lzma.LZMAError,
        zlib.error,
        EOFError,
    ) as error:
        raise ToolkitConfigError(f"Could not extract CUDA Toolkit archive {archive}: {error}") from error
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ToolkitConfigError(f"CUDA Toolkit archive {archive} must contain one top-level directory")
    return roots[0]


def _remove_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def merge_component(source: Path, destination: Path) -> None:
    """Merge a component so later archives replace earlier entries."""
    destination.mkdir(parents=True, exist_ok=True)
    for source_entry in source.iterdir():
        target = destination / source_entry.name
        if source_entry.is_dir() and not source_entry.is_symlink():
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                _remove_path(target)
            target.mkdir(exist_ok=True)
            merge_component(source_entry, target)
        else:
            if target.exists() or target.is_symlink():
                _remove_path(target)
            if source_entry.is_symlink():
                os.symlink(
                    os.readlink(source_entry),
                    target,
                    target_is_directory=source_entry.is_dir(),
                )
            else:
                shutil.copy2(source_entry, target)


def install_bundle(
    bundle: Bundle,
    cuda_path: Path,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download and assemble one Toolkit bundle."""
    installed = Path(os.path.abspath(cuda_path))
    installed.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{installed.name}.staging-",
        dir=installed.parent,
        ignore_cleanup_errors=True,
    ) as temporary:
        staging = Path(temporary)
        toolkit = staging / "toolkit"
        toolkit.mkdir()
        for index, record in enumerate(bundle.archives):
            archive = download_archive(
                record,
                staging / "downloads" / str(index),
                opener=opener,
                sleep=sleep,
            )
            component = extract_archive(
                archive,
                staging / "components" / str(index),
            )
            merge_component(component, toolkit)

        library = toolkit / "lib"
        if bundle.platform.startswith("linux-") and library.exists():
            merge_component(library, toolkit / "lib64")
            shutil.rmtree(library)

        if installed.exists() or installed.is_symlink():
            _remove_path(installed)
        os.replace(toolkit, installed)
    return installed


def activate_toolkit(
    cuda_path: Path,
    platform: str,
    github_env: Path | None = None,
    github_path: Path | None = None,
) -> Path:
    """Validate and publish one assembled Toolkit."""
    if (github_env is None) != (github_path is None):
        raise ToolkitConfigError("--github-env and --github-path must be provided together")
    installed = Path(os.path.abspath(cuda_path))
    if not (installed / "include").is_dir():
        raise ToolkitConfigError("Installed CUDA Toolkit is missing include/")
    executable = "nvcc.exe" if platform.startswith("windows-") else "nvcc"
    nvcc = installed / "bin" / executable
    if not nvcc.is_file():
        raise ToolkitConfigError(f"Installed CUDA Toolkit is missing bin/{executable}")
    try:
        result = subprocess.run(
            [nvcc, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=NVCC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ToolkitConfigError(
            f"CUDA compiler {nvcc} --version timed out after {NVCC_TIMEOUT_SECONDS} seconds"
        ) from error
    except OSError as error:
        raise ToolkitConfigError(f"Could not run CUDA compiler {nvcc}: {error}") from error
    if result.returncode:
        output = (result.stderr or result.stdout).strip()
        detail = f": {output}" if output else ""
        raise ToolkitConfigError(f"CUDA compiler {nvcc} --version exited with status {result.returncode}{detail}")

    if github_env is not None and github_path is not None:
        environment = f"WARP_CUDA_PATH={installed}\nCUDA_HOME={installed}\nCUDA_PATH={installed}\n"
        if platform.startswith("linux-"):
            library_path = str(installed / "lib64")
            if existing := os.environ.get("LD_LIBRARY_PATH"):
                library_path += f":{existing}"
            environment += f"LD_LIBRARY_PATH={library_path}\n"
        with github_env.open("a", encoding="utf-8") as output:
            output.write(environment)
        with github_path.open("a", encoding="utf-8") as output:
            output.write(f"{installed / 'bin'}\n")
    return installed


def _package_metadata(bundle: Bundle) -> dict[str, object]:
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "cuda_version": bundle.version,
        "platform": bundle.platform,
        "bundle_digest": bundle.digest,
    }


def _validate_package_metadata(path: Path, bundle: Bundle) -> None:
    metadata = _load_json(path, "package metadata")
    expected = _package_metadata(bundle)
    _require_keys(metadata, set(expected), "package metadata")
    if (
        not isinstance(metadata["schema_version"], int)
        or isinstance(metadata["schema_version"], bool)
        or metadata != expected
    ):
        raise _error(
            "package metadata",
            f"expected {expected!r}, found {metadata!r}",
        )


def pack_toolkit(
    bundle: Bundle,
    cuda_path: Path,
    archive_path: Path,
) -> Path:
    """Package one assembled Toolkit."""
    installed = Path(os.path.abspath(cuda_path))
    archive = Path(os.path.abspath(archive_path))
    if not archive.name.endswith(".tar.xz"):
        raise ToolkitConfigError("CUDA Toolkit package must end in .tar.xz")
    if archive.is_relative_to(installed):
        raise ToolkitConfigError("CUDA Toolkit package cannot be inside the Toolkit directory")
    installed = activate_toolkit(installed, bundle.platform)
    archive.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = f"{PACKAGE_ROOT}/{PACKAGE_METADATA_NAME}"
    metadata = (json.dumps(_package_metadata(bundle), indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=archive.parent,
            prefix=f".{archive.name}.",
            delete=False,
        ) as output:
            temporary = Path(output.name)
        with tarfile.open(temporary, "w:xz") as stream:
            stream.add(
                installed,
                arcname=PACKAGE_ROOT,
                filter=lambda member: None if member.name == metadata_path else member,
            )
            info = tarfile.TarInfo(metadata_path)
            info.mode = 0o644
            info.size = len(metadata)
            stream.addfile(info, io.BytesIO(metadata))
        os.replace(temporary, archive)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return archive


def unpack_toolkit(
    bundle: Bundle,
    archive_path: Path,
    cuda_path: Path,
) -> Path:
    """Restore one packaged Toolkit."""
    archive = Path(os.path.abspath(archive_path))
    if not archive.name.endswith(".tar.xz"):
        raise ToolkitConfigError("CUDA Toolkit package must end in .tar.xz")
    installed = Path(os.path.abspath(cuda_path))
    installed.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{installed.name}.staging-",
        dir=installed.parent,
        ignore_cleanup_errors=True,
    ) as temporary:
        try:
            toolkit = extract_archive(
                archive,
                Path(temporary) / "package",
            )
        except OSError as error:
            raise ToolkitConfigError(f"Could not extract CUDA Toolkit package {archive}: {error}") from error
        if toolkit.name != PACKAGE_ROOT:
            raise ToolkitConfigError(f"CUDA Toolkit package root must be {PACKAGE_ROOT!r}")
        _validate_package_metadata(
            toolkit / PACKAGE_METADATA_NAME,
            bundle,
        )
        activate_toolkit(toolkit, bundle.platform)
        if installed.exists() or installed.is_symlink():
            _remove_path(installed)
        os.replace(toolkit, installed)
    return installed


def _write_github_outputs(
    bundle: Bundle,
    github_output: Path,
    runner_temp: Path,
) -> None:
    cuda_path = runner_temp / "warp-cuda-toolkits" / bundle.digest
    with github_output.open("a", encoding="utf-8") as output:
        output.write(f"bundle-digest={bundle.digest}\ncache-key={bundle.cache_key}\ncuda-path={cuda_path}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("update-lock")
    commands.add_parser("validate-lock")

    resolve = commands.add_parser("resolve")
    resolve.add_argument("--version", required=True)
    resolve.add_argument("--platform", required=True)
    resolve.add_argument("--github-output", type=Path)
    resolve.add_argument("--runner-temp", type=Path)

    for name in ("install", "pack", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("--version", required=True)
        command.add_argument("--platform", required=True)
        command.add_argument("--cuda-path", required=True, type=Path)
        if name != "install":
            command.add_argument("--archive", required=True, type=Path)

    activate = commands.add_parser("activate")
    activate.add_argument("--platform", required=True)
    activate.add_argument("--cuda-path", required=True, type=Path)
    activate.add_argument("--github-env", type=Path)
    activate.add_argument("--github-path", type=Path)

    args = parser.parse_args(argv)
    if args.command == "resolve" and ((args.github_output is None) != (args.runner_temp is None)):
        parser.error("--github-output and --runner-temp must be provided together")
    if args.command == "activate" and ((args.github_env is None) != (args.github_path is None)):
        parser.error("--github-env and --github-path must be provided together")
    return args


def _configured_bundle(version: str, platform: str) -> Bundle:
    return resolve_bundle(
        load_requirements(REQUIREMENTS_PATH),
        load_lock(LOCK_PATH),
        version,
        platform,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CUDA Toolkit helper."""
    args = parse_args(argv)
    try:
        if args.command == "update-lock":
            lock = update_lock()
            print(f"Wrote {len(lock.bundles)} CUDA Toolkit bundles")
        elif args.command == "validate-lock":
            lock = load_lock(LOCK_PATH)
            count = validate_lock(
                load_requirements(REQUIREMENTS_PATH),
                lock,
            )
            print(f"Validated {count} archives across {len(lock.bundles)} bundles")
        elif args.command == "resolve":
            bundle = _configured_bundle(args.version, args.platform)
            print(
                json.dumps(
                    {
                        "bundle_digest": bundle.digest,
                        "cache_key": bundle.cache_key,
                    },
                    sort_keys=True,
                )
            )
            if args.github_output is not None:
                _write_github_outputs(
                    bundle,
                    args.github_output,
                    args.runner_temp,
                )
        elif args.command == "install":
            bundle = _configured_bundle(args.version, args.platform)
            installed = install_bundle(bundle, args.cuda_path)
            print(f"Installed CUDA {bundle.version} for {bundle.platform} at {installed}")
        elif args.command == "pack":
            bundle = _configured_bundle(args.version, args.platform)
            archive = pack_toolkit(bundle, args.cuda_path, args.archive)
            print(f"Packed CUDA {bundle.version} for {bundle.platform} at {archive}")
        elif args.command == "unpack":
            bundle = _configured_bundle(args.version, args.platform)
            installed = unpack_toolkit(
                bundle,
                args.archive,
                args.cuda_path,
            )
            print(f"Unpacked CUDA {bundle.version} for {bundle.platform} at {installed}")
        else:
            installed = activate_toolkit(
                args.cuda_path,
                args.platform,
                args.github_env,
                args.github_path,
            )
            print(f"Activated CUDA Toolkit at {installed}")
        return 0
    except (
        ToolkitConfigError,
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
