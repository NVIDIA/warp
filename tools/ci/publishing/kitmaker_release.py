# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and monitor releases on the Kitmaker portal.

All configuration (portal URL, project ID, etc.) is read from environment
variables, with optional CLI argument overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlsplit

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CANONICAL_WHEEL_PLATFORMS = {
    "manylinux_2_28_x86_64",
    "manylinux_2_34_aarch64",
    "win_amd64",
    "macosx_11_0_arm64",
}
# Windows on Arm wheels are built by GitHub Actions and uploaded to Artifactory
# by hand, so publishing treats them as optional for now, with no set end date.
# After this workflow has worked for a few releases, consider moving win_arm64
# into CANONICAL_WHEEL_PLATFORMS so publishing refuses to proceed without it.
OPTIONAL_WHEEL_PLATFORMS = {"win_arm64"}
OPTIONAL_WHEEL_LOOKUP_TIMEOUT = 30
WARP_WHEEL_FILENAME_PATTERN = re.compile(r"^warp_lang-(?P<version>[^-]+)-py3-none-(?P<platform>[^.]+)\.whl$")


class KitmakerClient:
    """Client for the Kitmaker release API with retry and monitoring support."""

    def __init__(self, base_url: str, api_token: str, verify_ssl: bool = True):
        self.base_url = base_url
        self.verify_ssl = verify_ssl
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        })
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    def create_release(
        self,
        project_id,
        project_name,
        pic_email,
        wheels,
        upload=True,
        devzone_subdir=None,
        mirroring_strategy=None,
    ):
        """Create a release and return the release UUID."""
        url = f"{self.base_url}/projects/{project_id}/releases"
        payload = {
            "project_name": project_name,
            "payload": [
                {
                    "pic": pic_email,
                    "job_type": "wheel-release-job",
                    "url": wheel_url,
                    "upload": upload,
                    **({"mirroring_strategy": mirroring_strategy} if mirroring_strategy else {}),
                    **({"devzone_subdir": devzone_subdir} if devzone_subdir else {}),
                }
                for wheel_url in wheels
            ],
        }

        print("Creating release...")
        response = self.session.post(url, json=payload, verify=self.verify_ssl, timeout=60)

        print(f"\n--- API RESPONSE ---")
        print(f"HTTP Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
        print("-------------------\n")

        if response.status_code >= 400:
            print("--- REQUEST DETAILS (for debugging) ---")
            print(f"URL: {url}")
            print(f"Payload sent:")
            print(json.dumps(payload, indent=2))
            print("---------------------------------------\n")

        response.raise_for_status()

        data = response.json()
        release_uuid = data.get("release_uuid")
        if not release_uuid:
            raise ValueError(f"No release_uuid in response: {data}")

        print(f"Created release: {release_uuid}")
        return release_uuid

    def monitor_release(self, release_uuid, poll_interval=30, timeout=None):
        """Poll release status until completed, failed, or timeout."""
        print(f"\nMonitoring release progress (polling every {poll_interval}s)...")
        start_time = time.time()

        while True:
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Release monitoring exceeded {timeout}s timeout")

            try:
                url = f"{self.base_url}/status/{release_uuid}"
                response = self.session.get(url, verify=self.verify_ssl, timeout=30)
                response.raise_for_status()
                status_data = response.json()
                status = status_data.get("status", "unknown")

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Release status: {status}")

                # Log per-wheel job details if present
                jobs = status_data.get("jobs") or status_data.get("release_jobs")
                if jobs:
                    for job in jobs:
                        job_status = job.get("status", "unknown")
                        job_url = job.get("url", job.get("wheel_url", ""))
                        print(f"  {job_status}: {job_url}")

                if status == "completed":
                    print("Release completed successfully!")
                    return True
                elif status == "failed":
                    print(f"Release failed! Response: {json.dumps(status_data, indent=2)}")
                    return False

            except (requests.RequestException, ValueError) as e:
                print(f"Warning: Status check failed: {e}")
                print(f"Retrying in {poll_interval} seconds...")

            time.sleep(poll_interval)


def validate_canonical_wheel_urls(wheel_urls):
    """Validate the canonical Warp wheel URL set, plus at most one optional wheel."""
    required_count = len(CANONICAL_WHEEL_PLATFORMS)
    if not required_count <= len(wheel_urls) <= required_count + 1:
        raise ValueError(
            f"expected {required_count} canonical wheel URLs plus at most 1 optional wheel URL, found {len(wheel_urls)}"
        )

    platforms = []
    versions = set()
    errors = []
    for wheel_url in wheel_urls:
        filename = wheel_url.rsplit("/", 1)[-1]
        match = WARP_WHEEL_FILENAME_PATTERN.fullmatch(filename)
        if match is None:
            errors.append(f"invalid Warp wheel filename: {filename}")
            continue
        versions.add(match.group("version"))
        platforms.append(match.group("platform"))

    actual_platforms = set(platforms)
    missing_platforms = CANONICAL_WHEEL_PLATFORMS - actual_platforms
    unsupported_platforms = actual_platforms - CANONICAL_WHEEL_PLATFORMS - OPTIONAL_WHEEL_PLATFORMS
    duplicate_platforms = {platform for platform in actual_platforms if platforms.count(platform) > 1}
    if missing_platforms:
        errors.append(f"missing platforms: {', '.join(sorted(missing_platforms))}")
    if unsupported_platforms:
        errors.append(f"unsupported platforms: {', '.join(sorted(unsupported_platforms))}")
    if duplicate_platforms:
        errors.append(f"duplicate platforms: {', '.join(sorted(duplicate_platforms))}")
    if len(versions) > 1:
        errors.append(f"mixed versions: {', '.join(sorted(versions))}")
    if any("+" in version for version in versions):
        errors.append("canonical wheels must not use a local version suffix")
    if errors:
        raise ValueError("invalid canonical Warp wheel URL set:\n- " + "\n- ".join(errors))


def get_wheel_filename_match(wheel_url):
    """Return the filename match for a Warp wheel URL, or None if the filename is not a Warp wheel."""
    return WARP_WHEEL_FILENAME_PATTERN.fullmatch(wheel_url.rsplit("/", 1)[-1])


def get_https_origin(url):
    """Return a normalized HTTPS origin, or None for invalid URLs or URLs with credentials."""
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
        ):
            return parsed.scheme, parsed.hostname, parsed.port if parsed.port is not None else 443
    except ValueError:
        pass
    return None


def select_optional_woa_wheel(wheel_urls, *, artifact_token, artifact_base_url=None, verify_ssl=True):
    """Return a new wheel list that includes the optional Windows on Arm wheel if Artifactory has it.

    The wheel URLs must already pass ``validate_canonical_wheel_urls``. The input list is not
    modified. No lookup outcome raises: unless the wheel is found, the reason is logged and the
    returned list holds the same wheels as the input. Discovery requires a valid
    ``artifact_base_url`` with the same HTTPS origin as the candidate wheel URL.
    """
    if any(get_wheel_filename_match(url).group("platform") in OPTIONAL_WHEEL_PLATFORMS for url in wheel_urls):
        print("Windows on Arm wheel supplied explicitly; skipping lookup")
        return list(wheel_urls)

    parents = {url.rsplit("/", 1)[0] for url in wheel_urls}
    version = get_wheel_filename_match(wheel_urls[0]).group("version")
    candidate = f"{wheel_urls[0].rsplit('/', 1)[0]}/warp_lang-{version}-py3-none-win_arm64.whl"
    artifact_origin = get_https_origin(artifact_base_url or "")
    could_not_check = (
        "Warning: could not check for optional Windows on Arm wheel ({}); publishing without it. "
        "Publish it separately if it was uploaded."
    )
    if len(parents) != 1:
        message = could_not_check.format("canonical wheels are not in one directory")
    elif not artifact_token:
        message = could_not_check.format("ARTIFACTORY_SVC_ACCESS_TOKEN is not set")
    elif not artifact_base_url:
        message = could_not_check.format("ARTIFACTORY_BASE_URL is not set")
    elif artifact_origin is None:
        message = could_not_check.format("ARTIFACTORY_BASE_URL is not a valid HTTPS URL without credentials")
    elif get_https_origin(candidate) != artifact_origin:
        message = could_not_check.format("wheel URL does not match the ARTIFACTORY_BASE_URL HTTPS origin")
    else:
        request_kwargs = {
            "headers": {"Authorization": f"Bearer {artifact_token}"},
            "timeout": OPTIONAL_WHEEL_LOOKUP_TIMEOUT,
            "verify": verify_ssl,
            "allow_redirects": True,
        }
        try:
            status = requests.head(candidate, **request_kwargs).status_code
            if status == 405:
                # Some repositories reject HEAD. Stream the GET so the wheel body is never downloaded.
                with requests.get(candidate, stream=True, **request_kwargs) as response:
                    status = response.status_code
        except requests.RequestException as error:
            message = could_not_check.format(f"{type(error).__name__} for {candidate}")
        else:
            if status == 200:
                return list(wheel_urls) + [candidate]
            if status == 404:
                message = f"Optional Windows on Arm wheel not found: {candidate}"
            else:
                message = could_not_check.format(f"HTTP {status} for {candidate}")

    print(message)
    # Skipping is allowed only while win_arm64 is optional; see OPTIONAL_WHEEL_PLATFORMS.
    return list(wheel_urls)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create and monitor releases on the Kitmaker portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python kitmaker_release.py \\
    https://<host>/<path>/warp_lang-1.0.0-py3-none-manylinux_2_28_x86_64.whl \\
    https://<host>/<path>/warp_lang-1.0.0-py3-none-macosx_11_0_arm64.whl \\
    --dry-run

Environment Variables:
  KITMAKER_API_TOKEN      Required. API token for authentication.
  KITMAKER_BASE_URL       Required. Base URL for the Kitmaker API.
  KITMAKER_PROJECT_ID     Required. Kitmaker project ID.
  KITMAKER_PROJECT_NAME   Required. Kitmaker project name.
  KITMAKER_PIC_EMAIL      Required. Person-in-charge email address.
  ARTIFACTORY_BASE_URL    Optional. Trusted HTTPS base URL for optional wheel discovery.
                          Discovery is skipped unless the wheel URL has the same origin.
  ARTIFACTORY_SVC_ACCESS_TOKEN
                          Optional. Artifactory token used only to look for the
                          optional win_arm64 wheel with --require-canonical-wheel-set.
        """,
    )

    parser.add_argument("wheel_urls", nargs="+", metavar="WHEEL_URL",
                        help="HTTPS URL(s) to .whl file(s) (1-5 URLs)")
    parser.add_argument("--devzone-subdir", default=None,
                        help="Optional Devzone subdirectory prefix for uploads")
    parser.add_argument(
        "--mirroring-strategy",
        choices=("upload-both", "wheel-stub", "claim-project"),
        default=None,
        help="Optional Kitmaker publication strategy",
    )
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status checks (default: 30)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Max seconds to wait for completion (default: 3600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without uploading")
    parser.add_argument("--base-url", default=None,
                        help="Kitmaker API base URL (default: KITMAKER_BASE_URL env var)")
    parser.add_argument("--project-id", default=None,
                        help="Kitmaker project ID (default: KITMAKER_PROJECT_ID env var)")
    parser.add_argument("--project-name", default=None,
                        help="Kitmaker project name (default: KITMAKER_PROJECT_NAME env var)")
    parser.add_argument("--pic-email", default=None,
                        help="Person-in-charge email (default: KITMAKER_PIC_EMAIL env var)")
    parser.add_argument("--no-verify-ssl", action="store_true",
                        help="Disable SSL certificate verification")
    parser.add_argument(
        "--require-canonical-wheel-set",
        action="store_true",
        help=(
            "Require the complete canonical Warp wheel URL set. Also looks in Artifactory for an optional "
            "win_arm64 wheel next to the canonical wheels and publishes it if found"
        ),
    )

    args = parser.parse_args(argv)

    # Resolve config: CLI args take priority over env vars
    config = {
        "base_url": args.base_url or os.environ.get("KITMAKER_BASE_URL"),
        "project_id": args.project_id or os.environ.get("KITMAKER_PROJECT_ID"),
        "project_name": args.project_name or os.environ.get("KITMAKER_PROJECT_NAME"),
        "pic_email": args.pic_email or os.environ.get("KITMAKER_PIC_EMAIL"),
    }
    missing = [k for k, v in config.items() if not v]
    if missing:
        print(f"Error: Missing required configuration: {', '.join(missing)}")
        return 1

    api_token = os.environ.get("KITMAKER_API_TOKEN")
    if not api_token:
        print("Error: KITMAKER_API_TOKEN environment variable is not set")
        return 1

    client = KitmakerClient(config["base_url"], api_token, verify_ssl=not args.no_verify_ssl)

    try:
        if len(args.wheel_urls) > 5:
            print(f"Error: Too many wheel URLs ({len(args.wheel_urls)}). Maximum is 5.")
            return 1

        wheels = []
        for wheel_url in args.wheel_urls:
            if not wheel_url.startswith("https://") or not wheel_url.lower().endswith(".whl"):
                print(f"Error: Invalid wheel URL (must be https:// and end with .whl): {wheel_url}")
                return 1
            wheels.append(wheel_url)
        if args.require_canonical_wheel_set:
            try:
                validate_canonical_wheel_urls(wheels)
                wheels = select_optional_woa_wheel(
                    wheels,
                    artifact_token=os.environ.get("ARTIFACTORY_SVC_ACCESS_TOKEN"),
                    artifact_base_url=os.environ.get("ARTIFACTORY_BASE_URL"),
                    verify_ssl=not args.no_verify_ssl,
                )
                validate_canonical_wheel_urls(wheels)
            except ValueError as error:
                print(f"Error: {error}")
                return 1

        # Configuration summary
        print(f"\nProject: {config['project_name']} (ID: {config['project_id']})")
        print(f"PIC: {config['pic_email']}")
        print(f"Upload: {'No (dry-run)' if args.dry_run else 'Yes'}")
        if args.mirroring_strategy:
            print(f"Mirroring strategy: {args.mirroring_strategy}")
        if args.devzone_subdir:
            print(f"Devzone subdir: {args.devzone_subdir}")
        print(f"Timeout: {args.timeout}s | Poll interval: {args.poll_interval}s")
        if args.require_canonical_wheel_set:
            has_woa = any(
                get_wheel_filename_match(url).group("platform") in OPTIONAL_WHEEL_PLATFORMS for url in wheels
            )
            print("Windows on Arm: included" if has_woa else "Windows on Arm: not included (optional)")
        print(f"Selected wheels for Kitmaker ({len(wheels)}):")
        for url in wheels:
            print(f"  {url}")

        release_uuid = client.create_release(
            project_id=config["project_id"],
            project_name=config["project_name"],
            pic_email=config["pic_email"],
            wheels=wheels,
            upload=not args.dry_run,
            devzone_subdir=args.devzone_subdir,
            mirroring_strategy=args.mirroring_strategy,
        )

        success = client.monitor_release(
            release_uuid,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
        return 0 if success else 1

    except requests.HTTPError as e:
        print(f"\nAPI Error: {e}")
        if e.response is not None:
            print(f"Response: {e.response.text}")
        return 1
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
